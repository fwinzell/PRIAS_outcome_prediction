import os
import pandas as pd
import h5py
import torch
import numpy as np
import random
import math
import re
from torchvision import transforms
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from PRIAS.augmentation.augmentation import GaussianNoise2Features


# from prias_file_reader import PRIAS_Data_Object


class PRIAS_Generic_Dataset(Dataset):
    """
    Generic dataset class for PRIAS feature/patch data
    Arguments:
        path_to_dir: Path to directory containing h5 files with features
        xls_path: Path to excel file containing patient ids and labels
        patient_data_obj: PRIAS_Data_Object, contains all patietn data, wsi names, etc. (see prias_file_reader.py)
        xls_sheet_name: Sheet name in excel file containing patient ids and labels, separate sheets for train and test
        shuffle: Shuffle order of patients in input
        seed: Random seed, set to fixt number for reproducibility
        val_split: Fraction of data to use for validation
        use_last_visit: Use only last visit for each patient, this is a simpler approach and also less useful
        use_features: Use features instead of patches, should always be true?
        use_long_labels: Use longitudinal labels
    """
    def __init__(self,
                 path_to_dir,
                 xls_path,
                 patient_data_obj,
                 xls_sheet_name=0,
                 shuffle=True,
                 seed=1,
                 val_split=0.1,
                 p_augmentation=(0.25, 0.25),  # (drop, gauss)
                 use_last_visit=True,
                 use_features=False,
                 use_long_labels=False,
                 long_time_intervals=None,
                 survival_mode=False,
                 n_month_mode=False,
                 n_months=30,
                 include_factors=False,
                 gg_dataset=False,
                 top_k_features=0,
                 filter_old=False):

        self.use_last = use_last_visit
        self.base_dir = path_to_dir
        self.val_split = val_split
        self.shuffle = shuffle
        self.use_features = use_features
        self.long = use_long_labels
        self.survival_mode = survival_mode
        self.n_month_mode = n_month_mode
        self.n_months = n_months

        # Time-point intervals to use (in years since baseline):
        # Alt 1: [0-1, 1-2, 2-3, 3-5, 5+]
        # Alt 2: [0, 0-3, 3+]
        # (should not model 3+ as np.inf? It will always be nan for active patients,
        # and irrelevant for treated?)
        # Alt 3: [baseline, [0, 3, 6]]
        if (self.long or self.survival_mode) and long_time_intervals is None:
            self.long_time_intervals = np.array([0, 3, 6]) * 365
        else:
            self.long_time_intervals = long_time_intervals   

        self.incl_rf = include_factors
        self.p_drop = p_augmentation[0]
        self.p_gauss = p_augmentation[1]
        self.top_k_features = top_k_features
        self.filter_old = filter_old
        random.seed(seed)
        torch.manual_seed(seed)
        # self.eval = eval
        # self.kwargs = kwargs

        # self.h5_files = [file for file in os.listdir(path_to_dir) if file.endswith('.h5')]

        # Dataframe with patient ids and corresponding labels
        label_data = pd.read_excel(xls_path, sheet_name=xls_sheet_name)

        self.p_dict = {'patient_number': np.array(label_data['Patient number']),
                       'status': np.array(label_data['act0 treated1'])}

        self.n_patients = len(self.p_dict['patient_number'])

        self.data_dict = self.find_slides_and_labels(patient_data_obj)

        if self.use_features:
            if self.incl_rf:
                self.DatasetType = Feature_w_Factors_Dataset
            elif gg_dataset:
                self.DatasetType = PRIAS_Feature_GG_Dataset
            else:
                self.DatasetType = PRIAS_Feature_Dataset
        else:
            self.DatasetType = PRIAS_Patch_Dataset

    def find_slides_and_labels(self, data_obj):
        """
        Finds all slides and corresponding labels for each slide
        Returns a data dictionary with patient ids, slide names and labels
        Can choose between longitudinal labels or simple binary labels
         - Checks that for each visit there is a corresponding h5 file (check_h5_file)
         - Long labels: From current visit in each timepoint in intervals, 1 if treated, 0 if not treated,
                        nan if unknown (each label is a list)
         - Simple labels: 1 if treated, 0 if not treated before the next visit
         data_dict has 3 (or 5) keys. Each key has a list for with one entry for every valid visit
         if long labels, labels is a list of lists
        """

        data_dict = {'pid': [], 'wsi': [], 'label': [], 'psa': [], 'vol': []}

        # Longitudinal labels
        if self.long and not self.survival_mode:
            for i, num in enumerate(self.p_dict['patient_number']):
                # Patient WSIs on the form {'YYPZ XXXXX': [1,2,...], ...} YY = year, Z = hospital (M,L or K)
                pat_wsis = data_obj.get_patient_data(num)
                # Timepoints for each visit in days since baseline
                t = data_obj.get_biopsy_timeline(num)

                if self.incl_rf:
                    (psa, vol) = data_obj.get_psa_and_volume(num, all=True)

                intervals = np.append(self.long_time_intervals, np.inf) # add inf for outside interval prediciton
                for j, wsi in enumerate(pat_wsis):
                    if self.filter_old and int(wsi[:2]) < 11:
                        print(f"Skipping {wsi}, too old...")
                        continue
                    if self.check_h5_file(wsi):  # Check that h5 file exists
                        data_dict['pid'].append(num)
                        data_dict['wsi'].append(wsi)
                        delta = t[-1] - t[j]  # time between vist j and last visit, i.e. remaining known time frame
                        if self.p_dict['status'][i] == 1:
                            # treated
                            labels = [1, []]
                            h = np.heaviside(intervals - delta, 1)
                            labels[1] = h  #np.append(h, 1)   # append baseline risk as 1
                        else:
                            # active
                            labels = [0, []]
                            h = np.zeros(len(intervals))
                            h[np.heaviside(intervals - delta, 0) == 1] = np.nan  # Set unknown timepoints to nan
                            labels[1] = h  #np.append(0, h)   # append baseline risk as 0
                        data_dict['label'].append(labels)

                        if self.incl_rf:
                            data_dict['psa'].append(psa[j])
                            data_dict['vol'].append(vol[j])
                    else:
                        print(f"Warning: {wsi} for patient {num} does not exist")

        # Survival labels, Bins and indicators
        elif self.survival_mode:
            for i, num in enumerate(self.p_dict['patient_number']):
                # Patient WSIs on the form {'YYPZ XXXXX': [1,2,...], ...} YY = year, Z = hospital (M,L or K)
                pat_wsis = data_obj.get_patient_data(num)
                # Timepoints for each visit in days since baseline
                t = data_obj.get_biopsy_timeline(num, in_months=True)

                if self.incl_rf:
                    (psa, vol) = data_obj.get_psa_and_volume(num, all=True)

                # Time-point intervals to use:
                intervals = torch.tensor(self.long_time_intervals)  
                for j, wsi in enumerate(pat_wsis):
                    #if self.filter_old and int(wsi[:2]) < 11:
                    #    print(f"Skipping {wsi}, too old...")
                    #    continue
                    if self.check_h5_file(wsi):  # Check that h5 file exists
                        data_dict['pid'].append(num)
                        data_dict['wsi'].append(wsi)
                        delta = t[-1] - t[j]  # time between vist j and last visit, i.e. remaining known time frame
                        indicator = self.p_dict['status'][i]
                        #event_bin = torch.bucketize(delta, intervals, right=True)
                        labels = [indicator, delta]
                        data_dict['label'].append(labels)

                        if self.incl_rf:
                            data_dict['psa'].append(psa[j])
                            data_dict['vol'].append(vol[j])
                    else:
                        print(f"Warning: {wsi} for patient {num} does not exist")

        elif self.n_month_mode:
            n_months = self.n_months #30 # Can change this later
            data_dict['delta'] = []
            data_dict['time'] = []
            for i, num in enumerate(self.p_dict['patient_number']):
                pat_wsis = data_obj.get_patient_data(num)
                t = data_obj.get_biopsy_timeline(num, in_months=True)

                if self.incl_rf:
                    (psa, vol) = data_obj.get_psa_and_volume(num, all=True)

                for j, wsi in enumerate(pat_wsis):
                    if self.filter_old and int(wsi[:2]) < 11:
                        print(f"Skipping {wsi}, too old...")
                        continue
                    if self.p_dict['status'][i] == 1:
                        if t[-1] - t[j] < n_months:
                            if self.check_h5_file(wsi):
                                data_dict['pid'].append(num)
                                data_dict['wsi'].append(wsi)
                                data_dict['label'].append(self.p_dict['status'][i])
                                data_dict['delta'].append(t[-1] - t[j])
                                data_dict['time'].append(t[j])

                                if self.incl_rf:
                                    data_dict['psa'].append(psa[j])
                                    data_dict['vol'].append(vol[j])
                            else:
                                print(f"Warning: {wsi} for patient {num} does not exist")
                    else:
                        # Do not include last visit for active patients
                        if j < len(pat_wsis) - 1:
                            if self.check_h5_file(wsi):
                                data_dict['pid'].append(num)
                                data_dict['wsi'].append(wsi)
                                data_dict['label'].append(self.p_dict['status'][i])
                                data_dict['delta'].append(t[-1] - t[j])
                                data_dict['time'].append(t[j])

                                if self.incl_rf:
                                    data_dict['psa'].append(psa[j])
                                    data_dict['vol'].append(vol[j])
                            else:
                                print(f"Warning: {wsi} for patient {num} does not exist")


        # Simple binary label
        else:
            for i, num in enumerate(self.p_dict['patient_number']):
                pat_wsis = data_obj.get_patient_data(num)

                if self.incl_rf:
                    (psa, vol) = data_obj.get_psa_and_volume(num, all=True)

                if self.use_last or self.p_dict['status'][i] == 1:
                    # Use only last vist
                    wsi = list(pat_wsis.keys())[-1]

                    if self.filter_old and int(wsi[:2]) < 11:
                        print(f"Skipping {wsi}, too old...")
                        continue
                    if self.check_h5_file(wsi):
                        data_dict['pid'].append(num)
                        data_dict['wsi'].append(wsi)
                        data_dict['label'].append(self.p_dict['status'][i])

                        if self.incl_rf:
                            data_dict['psa'].append(psa[-1])
                            data_dict['vol'].append(vol[-1])

                    else:
                        print(f"Warning: {wsi} for patient {num} does not exist")
                else:
                    # Use all visits
                    for j, wsi in enumerate(pat_wsis):
                        if self.filter_old and int(wsi[:2]) < 11:
                            print(f"Skipping {wsi}, too old...")
                            continue
                        if self.check_h5_file(wsi):
                            data_dict['pid'].append(num)
                            data_dict['wsi'].append(wsi)
                            data_dict['label'].append(self.p_dict['status'][i])

                            if self.incl_rf:
                                data_dict['psa'].append(psa[j])
                                data_dict['vol'].append(vol[j])
                        else:
                            print(f"Warning: {wsi} for patient {num} does not exist")

        return data_dict

    def check_h5_file(self, slide):
        """
        Check that feature file exists for slide
        """
        h5_fname = os.path.join(self.base_dir, f"{slide}.h5")
        return os.path.exists(h5_fname)

    def return_splits(self, set_split=None, shuffle=True, return_coords=False):
        """
        Create splits for training and validation. This method returns the datasets used in train.py
        Can choose between patches and features (use features)
        """
        if set_split is not None:
            self.val_split = set_split

        if self.val_split > 0:
            if self.shuffle:
                idxs = torch.randperm(self.n_patients)
            else:
                idxs = range(self.n_patients)
            split = int(self.val_split * self.n_patients)
            tr_ids = np.array(self.p_dict['patient_number'])[idxs[split:]]
            val_ids = np.array(self.p_dict['patient_number'])[idxs[:split]]

            tr_idxs = [i for i, pid in enumerate(self.data_dict['pid']) if pid in tr_ids]
            val_idxs = [i for i, pid in enumerate(self.data_dict['pid']) if pid in val_ids]

            tr_dict = {'patient_number': np.array(self.data_dict['pid'])[tr_idxs],
                       'labels': [self.data_dict['label'][ii] for ii in tr_idxs],
                       'slides': np.array(self.data_dict['wsi'])[tr_idxs]}

            val_dict = {'patient_number': np.array(self.data_dict['pid'])[val_idxs],
                        'labels': [self.data_dict['label'][ii] for ii in val_idxs],
                        'slides': np.array(self.data_dict['wsi'])[val_idxs]}
            
            if "delta" in self.data_dict.keys():
                tr_dict['delta'] = np.array(self.data_dict['delta'])[tr_idxs]
                val_dict['delta'] = np.array(self.data_dict['delta'])[val_idxs]

            if self.incl_rf:
                tr_dict['psa'] = np.array(self.data_dict['psa'])[tr_idxs]
                tr_dict['vol'] = np.array(self.data_dict['vol'])[tr_idxs]
                val_dict['psa'] = np.array(self.data_dict['psa'])[val_idxs]
                val_dict['vol'] = np.array(self.data_dict['vol'])[val_idxs]

            tr_split = self.DatasetType(tr_dict, self.base_dir, train_mode=True,
                                        p_drop=self.p_drop, p_gauss=self.p_gauss,
                                        topk=self.top_k_features, shuffle=shuffle, return_coords=return_coords)
            val_split = self.DatasetType(val_dict, self.base_dir, train_mode=False,
                                          p_drop=self.p_drop, p_gauss=self.p_gauss,
                                          topk=self.top_k_features, shuffle=shuffle, return_coords=return_coords)

            """if self.use_features:
                if self.incl_rf:
                    tr_split = Feature_w_Factors_Dataset(tr_dict, self.base_dir, train_mode=True)
                    val_split = Feature_w_Factors_Dataset(val_dict, self.base_dir, train_mode=False)
                else:
                    tr_split = PRIAS_Feature_Dataset(tr_dict, self.base_dir, train_mode=True, p_gauss=self.p_gauss,
                                                     p_drop=self.p_drop, topk=self.top_k_features, shuffle=shuffle)
                    val_split = PRIAS_Feature_Dataset(val_dict, self.base_dir, train_mode=False,
                                                      topk=self.top_k_features, shuffle=shuffle)
            else:
                tr_split = PRIAS_Patch_Dataset(tr_dict, self.base_dir, use_last=self.use_last, shuffle=shuffle)
                val_split = PRIAS_Patch_Dataset(val_dict, self.base_dir, use_last=self.use_last, shuffle=shuffle)"""
            return tr_split, val_split
        else:
            """
            If split = 0 return all data in non-training mode (no augmentation). Used in testing
            """
            if self.shuffle:
                idxs = torch.randperm(self.n_patients)
            else:
                idxs = range(self.n_patients)
            ids = np.array(self.p_dict['patient_number'])[idxs]

            p_idxs = [i for i, pid in enumerate(self.data_dict['pid']) if pid in ids]

            ret_dict = {'patient_number': np.array(self.data_dict['pid'])[p_idxs],
                       'labels': [self.data_dict['label'][ii] for ii in p_idxs],
                       'slides': np.array(self.data_dict['wsi'])[p_idxs]}
            
            if "delta" in self.data_dict.keys():
                ret_dict['delta'] = np.array(self.data_dict['delta'])[p_idxs]
            
            # This is added for the Generic all method, used primarily for the Cox evaluation
            if 'time' in self.data_dict.keys():
                ret_dict['time'] = np.array(self.data_dict['time'])[p_idxs]
            
            if self.incl_rf:
                ret_dict['psa'] = np.array(self.data_dict['psa'])[p_idxs]
                ret_dict['vol'] = np.array(self.data_dict['vol'])[p_idxs]


            ret_set = self.DatasetType(ret_dict, self.base_dir, train_mode=False,
                                        p_drop=self.p_drop, p_gauss=self.p_gauss,
                                        topk=self.top_k_features, shuffle=shuffle, return_coords=return_coords)

            """if self.use_features:
                if self.incl_rf:
                    ret_set = Feature_w_Factors_Dataset(ret_dict, self.base_dir, train_mode=False)
                else:
                    ret_set = PRIAS_Feature_Dataset(ret_dict, self.base_dir, train_mode=False, shuffle=shuffle)
            else:
                ret_set = PRIAS_Patch_Dataset(ret_dict, self.base_dir, use_last=self.use_last, shuffle=shuffle)"""
            return ret_set, None

    def get_slide_dict(self):
        slide_dict = {'patient_number': np.array(self.data_dict['pid']),
                       'labels': self.data_dict['label'],
                       'slides': np.array(self.data_dict['wsi'])}
        return slide_dict

    def __len__(self):
        """
        Length of dataset = number of patients
        """
        return len(self.data_dict['pid'])

    def class_weights(self):
        if self.long:
            baseline = list(zip(*self.data_dict['label']))[0]
            weights = 1 - np.unique(baseline, return_counts=True)[1] / len(self)
            return weights
        else:
            classes = np.array(self.data_dict['label'])
            weights = 1 - np.unique(classes, return_counts=True)[1]/len(self)
            return weights
        
    def get_patient_ids(self):
        return self.data_dict['pid']

    def __getitem__(self, item):
        return None


class PRIAS_Patch_Dataset(Dataset):
    """
    This is not updated and will likely not work
    """
    def __init__(self,
                 data_dict,
                 base_dir,
                 max_batch_size=32,
                 **kwargs):

        super().__init__()
        self.data_dict = data_dict
        self.base_dir = base_dir
        self.max_batch_size = max_batch_size

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data_dict['patient_number'])

    def __getitem__(self, idx):
        pid = self.data_dict['patient_number'][idx]
        label = self.data_dict['label'][idx]
        slide = self.data_dict['slides'][idx]

        with h5py.File(os.path.join(self.base_dir, f"{slide}.h5"), 'r') as h5df_file:
            K = h5df_file["imgs"].shape[0]
            img_idx = torch.randperm(K)[:self.max_batch_size]
            if torch.is_tensor(h5df_file["imgs"][0]):
                imgs = h5df_file["imgs"][:]
            else:
                imgs = [self.transform(img).unsqueeze(0) for img in h5df_file["imgs"][:]]
                imgs = torch.cat(imgs[:], dim=0)
            return pid, imgs[img_idx], label


class PRIAS_Feature_Dataset(Dataset):
    """
    PRIAS dataset class for features
    Arguments:
        data_dict: Dictionary with patient ids, slide names and labels
        base_dir: Path to directory containing h5 files with features
        train_mode: If true, use data augmentation
        p_drop: Dropout probability
        p_gauss: Gaussian noise probability
        shuffle: Shuffle order of patients in input, used in _shuffle()
        return_coords: whether coordinates of patches with wsi number should be returned
    Returns:
        pid: Patient id
        feats: Features from h5 file
        label: Label for patient
    """
    def __init__(self,
                 data_dict,
                 base_dir,
                 train_mode,
                 p_drop=0.25,
                 p_gauss=0.25,
                 shuffle=True,
                 topk=0,
                 return_coords=False,
                 return_segmentation=False,
                 random_sampling=False,
                 **kwargs):

        super().__init__()
        self.data_dict = data_dict
        if shuffle:
            self._shuffle()
        self.base_dir = base_dir
        self.train = train_mode
        self.topk = topk
        self.return_coords = return_coords # return coordinates with wsi number included
        self.return_seg = return_segmentation
        self.random_sampling = random_sampling

        if train_mode:
            self.dropout = torch.nn.Dropout(p=p_drop)
            self.gaussian = GaussianNoise2Features(p=p_gauss)

    def get_weights(self):
        count = Counter(self.data_dict['labels'])
        weights = np.array(count.values()) / len(self)
        return weights

    def _shuffle(self):
        import random
        indxs = np.arange(len(self.data_dict['labels']))
        random.shuffle(indxs)

        for key in self.data_dict.keys():
            if key == 'labels':
                self.data_dict[key] = [self.data_dict[key][i] for i in indxs]
            else:
                self.data_dict[key] = np.array(self.data_dict[key])[indxs]

        #temp = list(zip(self.data_dict['labels'], self.data_dict['patient_number'], self.data_dict['slides']))
        #random.shuffle(temp)
        #self.data_dict['labels'], self.data_dict['patient_number'], self.data_dict['slides'] = zip(*temp)
        #for key in self.data_dict.keys(): self.data_dict[key] = list(self.data_dict[key])

    def __len__(self):
        return len(self.data_dict['patient_number'])
    
    def get_patient_ids(self):
        return self.data_dict['patient_number']

    def __getitem__(self, idx):
        pid = self.data_dict['patient_number'][idx]
        label = self.data_dict['labels'][idx]
        slide = self.data_dict['slides'][idx]
        
        delta = self.data_dict['delta'][idx] if 'delta' in self.data_dict.keys() else None
        time = self.data_dict['time'][idx] if 'time' in self.data_dict.keys() else None

        with h5py.File(os.path.join(self.base_dir, f"{slide}.h5"), 'r') as h5df_file:
            runs = [name for name in h5df_file.keys() if re.fullmatch(r'features(_\d+)?$', name)]
            dataset = random.choice(runs)

            if torch.is_tensor(h5df_file[dataset][0]):
                feats = h5df_file[dataset][:]
            else:
                feats = torch.Tensor(h5df_file[dataset][:])

            if self.random_sampling:
                feats = feats[:, torch.randperm(feats.shape[1]), :]

            if self.topk > 0:
                feats = feats[:, :self.topk, :,]

            if self.train:
                feats = self.gaussian(feats)
                feats = self.dropout(feats)

            if feats.numel() == 0:
                print(f"Warning: {slide} has no features")
                feats = torch.zeros((self.topk, 1, 1))

            try:
                coords = np.array(h5df_file["coords"])
                ret_dict = {'wsi_name': slide, 'coords': coords, 'time': time, 'delta': delta}
                if self.return_seg:
                    try:
                        seg = np.array(h5df_file["tissue_fraction"])
                        ret_dict['tissues'] = seg
                    except KeyError:
                        pass
            except KeyError:
                ret_dict = {'wsi_name': slide}
        
            if self.return_coords or self.return_seg:
                return pid, feats, label, ret_dict
            else:
                return pid, feats, label


class Feature_w_Factors_Dataset(PRIAS_Feature_Dataset):
    def __init__(self,
                 data_dict,
                 base_dir,
                 train_mode,
                 p_drop=0.25,
                 p_gauss=0.25,
                 shuffle=True,
                 **kwargs):

        super().__init__(data_dict, base_dir, train_mode, p_drop, p_gauss, shuffle, **kwargs)

    def _shuffle(self):
        import random
        temp = list(zip(self.data_dict['labels'], self.data_dict['patient_number'], self.data_dict['slides'],
                        self.data_dict['psa'], self.data_dict['vol']))
        random.shuffle(temp)
        (self.data_dict['labels'], self.data_dict['patient_number'], self.data_dict['slides'],
         self.data_dict['psa'], self.data_dict['vol']) = zip(*temp)
        for key in self.data_dict.keys(): self.data_dict[key] = list(self.data_dict[key])

    def __getitem__(self, idx):
        pid = self.data_dict['patient_number'][idx]
        label = self.data_dict['labels'][idx]
        slide = self.data_dict['slides'][idx]

        factors = (self.data_dict['psa'][idx], self.data_dict['vol'][idx])

        with h5py.File(os.path.join(self.base_dir, f"{slide}.h5"), 'r') as h5df_file:
            if torch.is_tensor(h5df_file["features"][0]):
                feats = h5df_file["features"][:]
            else:
                feats = torch.Tensor(h5df_file["features"][:])

            if self.train:
                feats = self.gaussian(feats)
                feats = self.dropout(feats)
            return pid, feats, label, factors



class Simple_Feature_Dataset(Dataset):
    """
    Simple features dataset class used in inference and tests
    This does not work properly for some reason, use Generic instead with return_splits(0)
    """
    def __init__(self, base_dir, data_frame, label_header='act0 treated1'):
        self.base_dir = base_dir
        self.df = data_frame
        self.lh = label_header
        self.fnames = self._select_h5files()

    def _select_h5files(self):
        all_files = [f for f in os.listdir(self.base_dir) if f.endswith(".h5")]
        fnames = []
        for f in all_files:
            with h5py.File(os.path.join(self.base_dir, f), 'r') as h5df_file:
                pid = h5df_file.attrs["patient"]
                if pid in list(self.df["Patient number"]):
                    fnames.append(f)
        return fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        h5_path = os.path.join(self.base_dir, self.fnames[idx])
        with h5py.File(h5_path, 'r') as h5df_file:
            # pid = h5df_file["patient"]
            pid = h5df_file.attrs["patient"]
            label = self.df[self.lh][list(self.df['Patient number']).index(pid)]
            if torch.is_tensor(h5df_file["features"][0]):
                feats = h5df_file["features"][:]
            else:
                feats = torch.Tensor(h5df_file["features"][:])
            return pid, feats, label, h5_path


class PRIAS_Feature_GG_Dataset(PRIAS_Feature_Dataset):
    """
    PRIAS dataset class for features with GG scores
    Arguments:
        data_dict: Dictionary with patient ids, slide names and labels
        base_dir: Path to directory containing h5 files with features
        train_mode: If true, use data augmentation
        p_drop: Dropout probability
        p_gauss: Gaussian noise probability
        shuffle: Shuffle order of patients in input, used in _shuffle()
        return_coords: whether coordinates of patches with wsi number should be returned
    Returns:
        pid: Patient id
        feats: Features from h5 file
        label: Label for patient
    """
    def __init__(self,
                 data_dict,
                 base_dir,
                 train_mode,
                 p_drop=0.25,
                 p_gauss=0.25,
                 shuffle=True,
                 topk=256,
                 return_coords=False,
                 random_sampling=False,
                 **kwargs):

        super().__init__(data_dict, base_dir, train_mode, p_drop, p_gauss, shuffle, topk=topk, 
                         return_coords=return_coords, return_segmentation=False, 
                         random_sampling=random_sampling)
        
    def __getitem__(self, idx):
        pid = self.data_dict['patient_number'][idx]
        label = self.data_dict['labels'][idx]
        slide = self.data_dict['slides'][idx]

        delta = self.data_dict['delta'][idx] if 'delta' in self.data_dict.keys() else None
        time = self.data_dict['time'][idx] if 'time' in self.data_dict.keys() else None


        with h5py.File(os.path.join(self.base_dir, f"{slide}.h5"), 'r') as h5df_file:
            runs = [name for name in h5df_file.keys() if re.fullmatch(r'features(_\d+)?$', name)]
            dataset = random.choice(runs)

            if torch.is_tensor(h5df_file[dataset][0]):
                feats = h5df_file[dataset][:]
            else:
                feats = torch.Tensor(h5df_file[dataset][:])

            gg_scores = h5df_file["gg_scores"][:]
            order = np.argsort(np.squeeze(-gg_scores)) # Sort by GG score, descending
            feats = feats.squeeze()[order, :]

            if self.topk > 0:
                feats = feats[:self.topk, :,]
            else:
                print("Warning: No topk features specified, using all features")
                
            if self.random_sampling:
                feats = feats[torch.randperm(feats.shape[0]), :]

            if self.train:
                feats = self.gaussian(feats)
                feats = self.dropout(feats)

            if feats.numel() == 0:
                print(f"Warning: {slide} has no features")
                feats = torch.zeros((self.topk, 1, 1))

            try:
                coords = np.array(h5df_file["coords"])
                ret_dict = {'wsi_name': slide, 'coords': coords, 'time': time, 'delta': delta}
                if self.return_seg:
                    try:
                        seg = np.array(h5df_file["tissue_fraction"])
                        ret_dict['tissues'] = seg
                    except KeyError:
                        pass
            except KeyError:
                ret_dict = {'wsi_name': slide}

            ret_dict['gg_probs'] = torch.sigmoid(torch.tensor(gg_scores[order]))
            if self.return_coords:
                return pid, feats, label, ret_dict
            else:
                return pid, feats, label 
            



class CrossValidation_Dataset(PRIAS_Generic_Dataset):
    """
    Extension of the generic dataset class for cross-validation
    Keeps track of folds and splits data accordingly
    """
    def __init__(self,
                 path_to_dir,
                 xls_path,
                 xls_sheet_name,
                 patient_data_obj,
                 n_folds,
                 p_augmentation=(0.25, 0.25),  # (drop, gauss)
                 seed=1,
                 use_last_visit=False,
                 drop_last=False,
                 shuffle=True,
                 use_features=True,
                 use_long_labels=False,
                 long_time_intervals=None,
                 survival_mode=False,
                 include_factors=False,
                 gg_dataset=False,
                 n_month_mode=False,
                 return_segmentation=False,
                 top_k_features=0,
                 filter_old=False,
                 random_sampling=True):        
        super().__init__(path_to_dir, xls_path, patient_data_obj, seed=seed, xls_sheet_name=xls_sheet_name,
                         shuffle=shuffle, use_last_visit=use_last_visit, use_features=use_features,
                         use_long_labels=use_long_labels, long_time_intervals=long_time_intervals, survival_mode=survival_mode, 
                         include_factors=include_factors, top_k_features=top_k_features, filter_old=filter_old,
                         n_month_mode=n_month_mode, gg_dataset=gg_dataset)
        self.labels = [0, 1]
        self.n_folds = n_folds
        self.drop_last = drop_last
        self.ret_seg = return_segmentation
        self.random_sampling = random_sampling  
        try:
            if len(p_augmentation) == 2:
                self.p_a = p_augmentation
            else:
                self.p_a = (p_augmentation[0], p_augmentation[0])
        except TypeError:
            self.p_a = (p_augmentation, p_augmentation)

        self.folds = self._create_folds()

    def _create_folds(self):
        # Creates n folds with equal ratios between different labels
        index = {}
        folds = {}
        fold_sz = []
        for y in self.labels:
            index[y] = np.where(np.array(self.p_dict['status']) == y)[0]
            if self.shuffle:
                random.shuffle(index[y])
            fold_sz.append(round(len(index[y]) / self.n_folds))

        for k in range(self.n_folds):
            folds[k] = []
            for j, yy in enumerate(self.labels):
                if k == self.n_folds - 1:
                    folds[k] += list(index[yy][k * fold_sz[j]:])
                else:
                    folds[k] += list(index[yy][k * fold_sz[j]:(k + 1) * fold_sz[j]])

        return folds

    def return_splits(self, K):

        tr_fold, val_fold = [], self.folds[K]
        for k_hat in range(self.n_folds):
            if k_hat != K:
                tr_fold += self.folds[k_hat]

        tr_ids = np.array(self.p_dict['patient_number'])[tr_fold]
        val_ids = np.array(self.p_dict['patient_number'])[val_fold]

        tr_idxs = [i for i, pid in enumerate(self.data_dict['pid']) if pid in tr_ids]
        val_idxs = [i for i, pid in enumerate(self.data_dict['pid']) if pid in val_ids]

        tr_dict = {'patient_number': np.array(self.data_dict['pid'])[tr_idxs],
                   'labels': [self.data_dict['label'][ii] for ii in tr_idxs],
                   'slides': np.array(self.data_dict['wsi'])[tr_idxs]}

        val_dict = {'patient_number': np.array(self.data_dict['pid'])[val_idxs],
                    'labels': [self.data_dict['label'][ii] for ii in val_idxs],
                    'slides': np.array(self.data_dict['wsi'])[val_idxs]}
        
        if "delta" in self.data_dict.keys():
            tr_dict['delta'] = np.array(self.data_dict['delta'])[tr_idxs]
            val_dict['delta'] = np.array(self.data_dict['delta'])[val_idxs]
        
        if self.incl_rf:
            tr_dict['psa'] = np.array(self.data_dict['psa'])[tr_idxs]
            tr_dict['vol'] = np.array(self.data_dict['vol'])[tr_idxs]
            val_dict['psa'] = np.array(self.data_dict['psa'])[val_idxs]
            val_dict['vol'] = np.array(self.data_dict['vol'])[val_idxs]

        tr_split = self.DatasetType(tr_dict, self.base_dir, train_mode=True,
                                    p_drop=self.p_a[0], p_gauss=self.p_a[1],
                                    topk=self.top_k_features, shuffle=self.shuffle, 
                                    return_segmentation=self.ret_seg, random_sampling=self.random_sampling)
        val_split = self.DatasetType(val_dict, self.base_dir, train_mode=False,
                                    p_drop=self.p_a[0], p_gauss=self.p_a[1],
                                    topk=self.top_k_features, shuffle=self.shuffle,
                                    return_segmentation=False, random_sampling=self.random_sampling)
        return tr_split, val_split
    

class PRIAS_Generic_All(PRIAS_Generic_Dataset):
    """
    This class is used to create a dataset including all patient visits, no matter the label.
    It can be used with only treated patients by setting the only_treated flag to True.
    Used for testing in its default setting
    """
    def __init__(self,
                 path_to_dir,
                 xls_path,
                 xls_sheet_name,
                 patient_data_obj,
                 val_split=0,
                 seed=1,
                 shuffle=True,
                 include_factors=False,
                 filter_old=True,
                 only_treated=False):
        self.only_treated = only_treated
        super().__init__(path_to_dir, xls_path, patient_data_obj, seed=seed, xls_sheet_name=xls_sheet_name,
                         shuffle=shuffle, use_features=True, val_split=val_split,
                         use_long_labels=False, include_factors=include_factors, filter_old=filter_old)
        
        self.pdo = patient_data_obj
          
    
    def find_slides_and_labels(self, data_obj):
        """
        Finds all slides and corresponding labels for each slide
        Returns a data dictionary with patient ids, slide names and labels
        Can choose between longitudinal labels or simple binary labels
         - Checks that for each visit there is a corresponding h5 file (check_h5_file)
         - Long labels: From current visit in each timepoint in intervals, 1 if treated, 0 if not treated,
                        nan if unknown (each label is a list)
         - Simple labels: 1 if treated, 0 if not treated before the next visit
         data_dict has 3 (or 5) keys. Each key has a list for with one entry for every valid visit
         if long labels, labels is a list of lists
        """

        data_dict = {'pid': [], 'wsi': [], 'label': [], 'time': [], 'psa': [], 'vol': []}

        for i, num in enumerate(self.p_dict['patient_number']):
                pat_wsis = data_obj.get_patient_data(num)
                t = data_obj.get_biopsy_timeline(num)
                t = np.round(np.array(t)/30.4) # convert to months

                if self.p_dict['status'][i] == 0 and self.only_treated:
                    continue

                if self.incl_rf:
                    (psa, vol) = data_obj.get_psa_and_volume(num, all=True)

                # Use all visits
                for j, wsi in enumerate(pat_wsis):
                    if self.filter_old and int(wsi[:2]) < 11:
                        print(f"Skipping {wsi}, too old...")
                        continue
                    if self.check_h5_file(wsi):
                        data_dict['pid'].append(num)
                        data_dict['wsi'].append(wsi)
                        data_dict['label'].append(self.p_dict['status'][i])
                        data_dict['time'].append(t[j])

                        if self.incl_rf:
                            data_dict['psa'].append(psa[j])
                            data_dict['vol'].append(vol[j])
                    else:
                        print(f"Warning: {wsi} for patient {num} does not exist")

        return data_dict


if __name__ == "__main__":
    """
    The main function is used for testing purposes
    """

    from prias_file_reader import PRIAS_Data_Object
    """
    # Test Generic
    import cv2

    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    dataset = PRIAS_Generic_Dataset(
        path_to_dir="/home/fi5666wi/PRIAS_data/features_imagenet",
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
        xls_sheet_name=3,
        use_last_visit=False,
        use_features=True,
        use_long_labels=True
    )

    w = dataset.class_weights()
    print(w)
    tr_dataset, val_dataset = dataset.return_splits(set_split=0.1)

    trloader = DataLoader(tr_dataset, batch_size=1)
    for i, (pid, inst, label) in enumerate(trloader):
        print(f"PID: {pid.item()}:  {label}")
        # x = np.squeeze(inst[0,0,:,:,:])
        # x = np.moveaxis(x.numpy(), source=0, destination=-1)
        # cv2.imshow('image 0', x)
        # cv2.waitKey(0)
        if i > 25:
            break
    """

    """
    # Test Simple
    device = torch.device("cuda:0")
    label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels_2.xlsx",
                               sheet_name=0)
    simple = Simple_Feature_Dataset(base_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet",
                                    data_frame=label_data)
    simloader = DataLoader(simple, batch_size=1)
    for i, (pid, inst, label, h5_path) in enumerate(simloader):
        print(f"{pid.item()}: {inst.shape}    {label.item()}")
        if inst.shape[-1] > 1:
            f = inst[:, :, :, 0, 0]
            f = torch.unsqueeze(f, -1)
            f = torch.unsqueeze(f, -1)
            with h5py.File(h5_path[0], 'r+') as h5df_file:
                data = h5df_file["features"]
                data[...] = f
                h5df_file.close()


    """

    #
    """
    # Test Cross-val
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    nfolds = 6
    cross_val_dataset = CrossValidation_Dataset(
        path_to_dir="/home/fi5666wi/PRIAS_data/features_uni_v2_augment_all",
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=1,
        n_folds=nfolds,
        seed=random.randint(0,100),
        use_features=True,
        use_long_labels=False,
        include_factors=False,
        filter_old=True,
    )

    for k in range(nfolds):
        print(f"############# Fold {k} ############")
        tr_data_fold, val_data_fold = cross_val_dataset.return_splits(K=k)
        val_loader = DataLoader(val_data_fold, batch_size=1)
        tr_loader = DataLoader(tr_data_fold, batch_size=1)
        print(f"{k}: {len(val_loader)}")
        pidde = []
        for i, (pid, inst, label) in enumerate(val_loader):
            pidde.append(pid.item())
            print(f"{pid.item()}: {inst.shape}    {label.item()}")    #{[rf.item() for rf in fac]}")
        print(pidde)

    """

    #"""
    # Test Features
    from prias_file_reader import PRIAS_Data_Object

    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    dataset = PRIAS_Generic_Dataset(
        path_to_dir="/home/fi5666wi/PRIAS_data/features_from_bengio",
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=0,
        use_last_visit=False,
        shuffle=True,
        use_features=True,
        use_long_labels=False,
        include_factors=False,
        n_month_mode=True,
        filter_old=True,
        gg_dataset=True
    )

    feature_set, _ = dataset.return_splits(set_split=0)
    feature_set.topk = 1000
    trloader = DataLoader(feature_set, batch_size=1)
    pids = []
    for i, (pid, inst, label) in enumerate(trloader):
        print(f"{pid.item()}: {inst.shape}    {label.item()}")    #{[rf.item() for rf in fac]}")
        #f = inst[:, :, :, 0, 0]
        #f = torch.unsqueeze(f, -1)
        #f = torch.unsqueeze(f, -1)
        pids.append(pid.item())
        #print(f"{pid}: {f.shape}    {label}")
    #"""
    #print(len(pids))
    #pids = np.unique(np.array(pids))
    #print(len(pids))
    #print(pids)

