import torch
import os
import numpy as np
import h5py
from PRIAS.protocol.wsi_loader import WSIDataset
from torch.utils.data import DataLoader
#from supervised.attention import get_wsi_attention
import torch.nn.functional as F
from skimage.io import imread
import pandas as pd
import cv2
from argparse import Namespace
import PIL

from PRIAS.prias_file_reader import PRIAS_Data_Object
from PRIAS.supervised.resnet import resnet50

PIL.Image.MAX_IMAGE_PIXELS = None

class PatientObject(object):
    def __init__(self, pid, base_dir ,wsi_dict, model, num_patches, save_path, only_last_visit=True, patch_sz=299):
        """
        Args:
            pid: str with patient id
            base_dir: path to dictionary with wsis
            wsi_dict: dictionary of wsi's from patient
            model: trained cancer detector
            num_patches: int number of patches to extract
        """

        self.pid = pid
        self.base_dir = base_dir
        # Need to generate proper wsi list here
        if only_last_visit:
            last = wsi_dict.popitem()
            self.wsi_dict = {last[0]: last[1]}
        else:
            self.wsi_dict = wsi_dict

        """ self.wsi_list = []
        for i, biopsy_id in enumerate(self.wsi_dict.keys()):
            if only_last_visit and i+1 != len(self.wsi_dict.keys()):
                continue
            self.wsi_list += [os.path.join(base_dir,
                                           "{}-{}_10x.png".format(biopsy_id, str(num))) for num in wsi_dict[biopsy_id]]
        """
        self.device = self.get_device()
        self.model = model.to(self.device)
        self.n = num_patches
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(save_path, exist_ok=True)

        self.patch_sz = patch_sz
        self.attn_scores = torch.Tensor
        self.patch_indxs = torch.Tensor
        self.pts = []
        self.hdf5_file = None

        #self.df = pd.DataFrame(data={'wsi_name': self.wsi_list,
        #             'num_patches': np.zeros(len(self.wsi_list))}, index=range(len(wsi_list)))

        self.wsi_list = [key + f"-{num}" for key in self.wsi_dict.keys() for num in self.wsi_dict[key]]
        self.df = pd.DataFrame(data={'wsi_id': self.wsi_list,
                                     'n_patches': np.zeros(len(self.wsi_list))},
                               index=range(len(self.wsi_list)))

    @staticmethod
    def get_device():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("running on: {}".format(device))
        return device

    def _get_wsi_scores_and_pts(self, wsi_path, overlap=30, batch_size=10):
        dataset = WSIDataset(wsi_path, patch_size=self.patch_sz, overlap=overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        atten_scores = torch.zeros(len(dataset))
        pts = torch.zeros((len(dataset), 2))
        for i, (data, coords) in enumerate(dataloader):
            with torch.no_grad():
                y_hat = F.sigmoid(self.model(data))
                prob = y_hat.squeeze()
                atten_scores[i*batch_size:(i+1)*batch_size] = prob
                pts[i*batch_size:(i+1)*batch_size, :] = coords

        return atten_scores, pts.int()

    def select_patches(self):
        for wsi_id in self.wsi_dict.keys():
            print(f'Processing: {wsi_id}')
            all_scores = []
            all_pts = []
            for num in self.wsi_dict[wsi_id]:
                print(f"WSI: {wsi_id}-{num}")
                wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")

                atten, pts = self._get_wsi_scores_and_pts(wsi_path)
                all_scores = all_scores + atten.tolist()

                wsi_ind = torch.ones((pts.shape[0], 1))*num
                pts_list = torch.cat((wsi_ind.int(), pts), dim=1)
                all_pts = all_pts + pts_list.tolist()

            scores = torch.Tensor(all_scores)
            self.pts = {wsi_id: all_pts}

            attn_scores, order = torch.sort(scores, descending=True)
            indxs = order[:self.n]
            patch_indxs, _ = torch.sort(indxs)
            self.attn_scores = {wsi_id: attn_scores}
            self.patch_indxs = {wsi_id: patch_indxs}
            print('Done')

    def save_patches(self):
        print("Saving patches")
        current_wsi = -1
        wsi = torch.zeros(1)
        for wsi_id in self.wsi_dict.keys():
            pts = self.pts[wsi_id]
            indxs = self.patch_indxs[wsi_id]
            for j in range(self.n):
                pt = pts[indxs[j]]
                (y,x) = pt[1:]
                if current_wsi != pt[0]:
                    current_wsi = pt[0]
                    wsi = imread(os.path.join(self.base_dir, f"{wsi_id}-{current_wsi}_10x.png"))
                patch = wsi[x:x + self.patch_sz, y:y + self.patch_sz, :]

                if self.hdf5_file is None:
                    self.hdf5_file = init_hdf5(patch, pt, self.save_path, self.pid, wsi_id, save_coord=True)
                else:
                    self._save_to_hdf5(patch, pt)

                self.df.loc[self.wsi_list.index(f"{wsi_id}-{current_wsi}"), 'n_patches'] += 1
            print(f"Saved {self.n} patches for {wsi_id}")
            self.hdf5_file = None


    def _save_to_hdf5(self, patch, pt):
        img_patch = np.array(patch)[np.newaxis, ...]
        img_shape = img_patch.shape

        file = h5py.File(self.hdf5_file, 'a')
        dset = file['imgs']
        dset.resize(len(dset) + img_shape[0], axis=0)
        dset[-img_shape[0]:] = img_patch

        if 'coords' in file:
            coord_dset = file['coords']
            coord_dset.resize(len(coord_dset) + img_shape[0], axis=0)
            coord_dset[-img_shape[0]:] = pt

        file.close()

    def write_df_to_csv(self, path):
        if os.path.exists(path):
            self.df.to_csv(path, mode='a', header=False)
        else:
            self.df.to_csv(path)

    def create_stitches(self, downsample_factor, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print("Creating stitches...")
        for j,wsi_id in enumerate(self.wsi_list):
            wsi_path = os.path.join(self.base_dir, f"{wsi_id}_10x.png")
            if os.path.exists(wsi_path):
                wsi = cv2.imread(wsi_path)
                dim = (int(wsi.shape[1] * downsample_factor), int(wsi.shape[0] * downsample_factor))
                wsi = cv2.resize(wsi, dim, interpolation=cv2.INTER_AREA)
                wsi_name, wsi_num = wsi_id.split("-")
                for pt in [list(self.pts.values())[0][i] for i in self.patch_indxs[wsi_name][:self.n]]:
                    if pt[0] == int(wsi_num):
                        x,y = np.array(pt[1:])*downsample_factor
                        x,y = int(x),int(y)
                        sz = int(self.patch_sz*downsample_factor)
                        cv2.rectangle(wsi, (x,y), (x+sz,y+sz), (0,0,255), thickness=1)
                cv2.imwrite(os.path.join(save_dir, wsi_id + ".png"), wsi)
        print("Done")


def init_hdf5(patch, pt, save_path, pid, wsi_id, save_coord=False):
    #x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name = tuple(first_patch.values())
    #x, y = pt[1:]
    file_path = os.path.join(save_path, wsi_id)+'.h5'
    file = h5py.File(file_path, "w")
    img_patch = np.array(patch)[np.newaxis,...]
    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('imgs',
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    #dset.attrs['patch_level'] = patch_level
    dset.attrs['patient'] = pid
    #dset.attrs['downsample'] = downsample
    #dset.attrs['level_dim'] = level_dim
    #dset.attrs['downsampled_level_dim'] = downsampled_level_dim

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 3), maxshape=(None, 3), chunks=(1, 3), dtype=np.int32)
        coord_dset[:] = pt

    file.close()
    return file_path


def load_resnet50():
    ckpt_path = os.path.join(os.getcwd(), 'supervised', 'saved_models',
                             'resnet50_2023-03-23', 'lightning_logs', 'version_0', 'checkpoints',
                             'last.ckpt')
    model_path = os.path.join(os.getcwd(), 'supervised', 'saved_models',
                             'resnet50_2023-04-28', 'version_0', 'last.pth')

    model = resnet50(Namespace(use_fcn=True, num_classes=2, binary=True))
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)
    return model #torch.load(model_path)


if __name__ == "__main__":
    wsi_dir = "/home/fi5666wi/Documents/PRIAS_data/wsis"
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    data = PRIAS_Data_Object(sheet_path, sheet_name=3)

    save_path = "/home/fi5666wi/Documents/PRIAS_data/patches_test" #os.path.join(pat_path, 'patches')
    model = load_resnet50()
    #wsi_list = [os.path.join(pat_path, 'images', wsi_path) for wsi_path in os.listdir(os.path.join(pat_path, 'images'))]

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    list_of_patients = pd.read_excel(xcl_path, sheet_name=2)['Patient number']

    for pid in list_of_patients:
        if pid < 35 or pid == 38:
            continue
        if pid > 50:
            break
        wsi_dict = data.get_patient_data(patient_no=pid)

        patient = PatientObject(pid=pid, base_dir=wsi_dir, wsi_dict=wsi_dict, model=model, num_patches=64, save_path=save_path)
        patient.select_patches()
        patient.save_patches()
        patient.write_df_to_csv(os.path.join(save_path, 'wsi_list.csv'))
        patient.create_stitches(downsample_factor=0.1, save_dir=os.path.join(save_path, 'stitches'))
    print('herå')



