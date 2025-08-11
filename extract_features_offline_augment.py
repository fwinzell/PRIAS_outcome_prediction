import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from PRIAS.protocol.wsi_loader import WSIDataset
import random
import re


# from supervised.attention import get_wsi_attention
import torch.nn.functional as F
from skimage.io import imread
import pandas as pd
import argparse
from argparse import Namespace
import PIL

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose, ToTensor, ColorJitter

from PRIAS.prias_file_reader import PRIAS_Data_Object
from PRIAS.supervised.resnet import resnet50
from PRIAS.supervised.densenet import densenet201
from PRIAS.extract_patches import PatientObject

#UNI
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

from UNI.uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

PIL.Image.MAX_IMAGE_PIXELS = None


class PatientFeaturesAugment(PatientObject):
    """
    Class for extracting features from WSIs for a given patient
    Builds on PatientObject class from extract_patches.py
    Arguments:
        pid: patient id
        base_dir: directory where WSIs are stored
        wsi_dict: dictionary with wsi names
        model: gleason grade model
        pt_model: feature extractor model, if different from model, otherwise None
        num_patches: number of patches to extract features from
        patch_overlap: overlap between patches
        fe_level: which level in the network to extract features from
        save_path: directory to save .h5 files
        last_visit: if True, only use last visit for each wsi
        patch_sz: size of patches as input to feature extractor (default 299, GG always uses 299x299 patches)
    """
    def __init__(self, pid, base_dir, wsi_dict, model, transform, num_patches, patch_overlap,
                 fe_level, save_path, last_visit=True, fe_patch_sz=299, use_uni=False, augment_params=[0.2, 0.2, 0.2, 0.2]):
        super(PatientFeaturesAugment, self).__init__(pid,
                                              base_dir,
                                              wsi_dict,
                                              model,
                                              num_patches,
                                              save_path,
                                              only_last_visit=last_visit)

        #self.save_path = save_path
        #self.feature_size = feature_size
        self.fe_patch_sz = fe_patch_sz
        self.overlap = patch_overlap
        self.level = fe_level
        #os.makedirs(save_path, exist_ok=True)
        
        # Use the same model for attention and feature extraction
        self.fe_model = model.to(self.device)

        self.transforms = Compose([
                                ToTensor(),
                                ColorJitter(brightness=augment_params[0], 
                                            contrast=augment_params[1], 
                                            saturation=augment_params[2], 
                                            hue=augment_params[3])])
        self.use_uni = use_uni

        if use_uni:
            self.fe = None
            self.nodes = None
            generic_out = self.fe_model(torch.zeros((1,3,fe_patch_sz,fe_patch_sz)).to(self.device))
            self.feature_shape = generic_out.shape[1:]
        else:
            self.nodes, _ = get_graph_node_names(self.fe_model)
            # return last layer features and final prediction
            self.fe = create_feature_extractor(self.fe_model, return_nodes=self.nodes[self.level:])
            generic_out = self.fe(torch.zeros((1,3,299,299)).to(self.device))
            self.feature_shape = generic_out[self.nodes[self.level]].shape[1:]

            print(f"{self.nodes[self.level]}: {list(self.feature_shape)}")

            
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))


    def _get_wsi_features(self, wsi_path, batch_size=10):
        """
        Extract features from a given wsi
        Uses WSIDataset class to split WSI into patches
        Returns:
            features: LxNx1x1 features (L = number of patches in wsi)
            pts: Lx2 coordinates of patches (top left corner)
        """
        # N: 2048 for resnet50
        dataset = WSIDataset(wsi_path, patch_size=self.patch_sz, overlap=self.overlap, transforms=self.transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        pts = torch.zeros((len(dataset), 2), device=self.device)
        features = torch.zeros(tuple([len(dataset), self.feature_shape[0], 1, 1]), device=self.device)
        for i, (data, coords) in enumerate(dataloader):
            data, coords = data.to(self.device), coords.to(self.device)

            with torch.no_grad():
                output = self.fe(data)
                h = output[self.nodes[self.level]]
                # For lower level features, do average pooling to get BxNx1x1 features
                # if dimension is 2 we only have BxN
                if h.dim() > 2 and h.shape[-1] > 1:
                    h = self.avgpool(h)   
                while h.dim() < 4:
                    h = h.unsqueeze(-1)
                pts[i*batch_size:(i+1)*batch_size, :] = coords
                features[i*batch_size:(i+1)*batch_size, :] = h

        return features.cpu(), pts.int().cpu()
    
    def _get_uni_features(self, wsi_path, batch_size=10):
        """
        Extract UNI features from a given wsi using built-in feature extractor
        Uses WSIDataset class to split WSI into patches
        Returns:
            features: LxNx1x1 features (L = number of patches in wsi)
            pts: Lx2 coordinates of patches (top left corner)
        """
        dataset = WSIDataset(wsi_path, patch_size=self.fe_patch_sz, overlap=self.overlap, transforms=self.transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        feature_dict = extract_patch_features_from_dataloader(self.fe_model, dataloader)

        # convert these to torch
        features = torch.Tensor(feature_dict['embeddings'])
        pts = torch.Tensor(feature_dict['labels']).type(torch.long)

        return features, pts
    
    def _check_if_features(self, n_iters):
        with h5py.File(self.hdf5_file, 'a') as file:  # Open in append mode to allow deletion
            runs = [name for name in file.keys() if re.fullmatch(r'features(_\d+)?$', name)]
            sizes = [file[ii][()].size for ii in runs]
            zero_count = 0
            for ii, size in zip(runs, sizes):
                if size == 0:
                    print(f"Removing empty feature dataset: {ii}")
                    del file[ii]
                    zero_count += 1
            # After deletion, check again if enough features exist
            if (len(runs) - zero_count) < n_iters:
                return False
            else:
                return True
            
    def check_coords(self):
        for wsi_id in self.wsi_dict.keys():
            print(f'Processing: {wsi_id}')
            all_pts = []
            
            no_coords = False
            if os.path.exists(os.path.join(save_path, wsi_id)+'.h5'):
                self.hdf5_file = os.path.join(save_path, wsi_id)+'.h5'
                with h5py.File(self.hdf5_file, 'r') as file:
                    coords = file["coords"][:]
                    if len(coords) == 0:
                        print(f"No coordinates found in {self.hdf5_file}")
                        no_coords = True

            if no_coords:
                for num in self.wsi_dict[wsi_id]:
                    wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")
                    if os.path.exists(wsi_path):
                        print(f"WSI: {wsi_id}-{num}")
                        if self.use_uni:
                            _, pts = self._get_uni_features(wsi_path)
                        else:
                            _, pts = self._get_wsi_scores_and_features(wsi_path)
                        
                        wsi_ind = torch.ones((pts.shape[0], 1))*num
                        pts_list = torch.cat((wsi_ind.int(), pts), dim=1)
                        all_pts = all_pts + pts_list.tolist()

                with h5py.File(self.hdf5_file, 'a') as file:
                    if 'coords' in file.keys():
                        del file['coords']
                    file.create_dataset('coords', data=all_pts)

    def select_and_save(self, n_iters=1):
        """
        Selects the n patches with the highest probability of cancer for each wsi
        Saves the features, coordinates and attention scores in a .h5 file from the n patches of each wsi
        Uses init_hdf5() to initialize the .h5 file
        Uses _get_wsi_scores_and_features() to extract features and probabilities
        Uses _save_to_hdf5() to save features, coordinates and probabilities to .h5 file
        Returns nothing
        """
        for i in range(n_iters):
            for wsi_id in self.wsi_dict.keys():
                print(f'Processing: {wsi_id}')
                all_pts = []
                all_features = []
                if os.path.exists(os.path.join(save_path, wsi_id)+'.h5'):
                    self.hdf5_file = os.path.join(save_path, wsi_id)+'.h5'
                    if self._check_if_features(n_iters):
                        print(f"Features already extracted for {wsi_id}")
                        continue

                for num in self.wsi_dict[wsi_id]:
                    wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")
                    if os.path.exists(wsi_path):
                        print(f"WSI: {wsi_id}-{num}")
                        if self.use_uni:
                            f, pts = self._get_uni_features(wsi_path)
                        else:
                            f, pts = self._get_wsi_scores_and_features(wsi_path)
                        
                        all_features = all_features + f.tolist()
                        wsi_ind = torch.ones((pts.shape[0], 1))*num
                        pts_list = torch.cat((wsi_ind.int(), pts), dim=1)
                        all_pts = all_pts + pts_list.tolist()

                if self.hdf5_file is None:
                    self.hdf5_file = init_hdf5(all_features, all_pts, self.save_path, self.pid, wsi_id, save_coord=True)
                else:
                    self._save_to_hdf5(all_features)

                self.pts = {wsi_id: all_pts}
                self.hdf5_file = None
                print('Done')

    def have_a_look(self):
        no_feature_list = []
        no_h5_list = []
        no_wsi_list = []
        for wsi_id in self.wsi_dict.keys():
            print(f'Loading: {wsi_id}')
            for num in self.wsi_dict[wsi_id]:
                wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")
                if not os.path.exists(wsi_path):
                    print(f"Warning: {wsi_id}-{num} does not exist")
                    no_wsi_list.append(f"{wsi_id}-{num}")
                    continue
                #wsi = imread(wsi_path)
                #print(f"WSI: {wsi_id}-{num}: {wsi.shape}")

            hdf5_file = os.path.join(save_path, wsi_id)+'.h5'   
            if os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, 'r') as file:
                    runs = [name for name in file.keys() if re.fullmatch(r'features(_\d+)?$', name)]
                    for ii in runs:
                        features = file[ii][()]
                        if features.size == 0:
                            print(f"Warning: {wsi_id} has no features")
                            no_feature_list.append(wsi_id)
            else:
                print(f"Warning: {hdf5_file} does not exist")
                no_h5_list.append(wsi_id)

        return no_feature_list, no_h5_list, no_wsi_list



    def _save_to_hdf5(self, features):
        """
        Save features and coordinates to respective dataset of initialized .h5 file
        """
        f_vec = np.array(features)[np.newaxis, ...]
        #shape = f_vec.shape

        if f_vec.size != 0:
            with h5py.File(self.hdf5_file, 'a') as file:
                existing = list(file.keys())
                i = 0
                while f'features_{i}' in existing:
                    i += 1
                file.create_dataset(f'features_{i}', data=f_vec)
        else:
            print(f"Warning: No features to save")
        

def init_hdf5(features, pt, save_path, pid, wsi_id, save_coord=True):
    """
    Initialize a .h5 file for saving features, coordinates and probabilities
    Documentations: https://docs.h5py.org/en/stable/high/file.html https://docs.h5py.org/en/stable/high/dataset.html
    Attributes:
        patient: patient id
    Datasets:
        features_x: features of patches
        coords: coordinates of patches (top left corner)
    """
    file_path = os.path.join(save_path, wsi_id)+'.h5'
    file = h5py.File(file_path, "w")
    file.attrs['patient'] = pid

    f_vec = np.array(features)[np.newaxis,...]
    if f_vec.size != 0:
        file.create_dataset('features_0', data=f_vec)
    else:
        print(f"Warning: No features to save")

    if save_coord:
        file.create_dataset('coords', data=pt)

    file.close()
    return file_path


def load_resnet50(base_dir='/home/fi5666wi/Python/PRIAS'):
    model_path = os.path.join(base_dir, 'supervised', 'saved_models',
                             'resnet50_2023-04-28', 'version_0', 'last.pth')

    model = resnet50(Namespace(use_fcn=True, num_classes=2, binary=True))
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)
    return model, 2048

def load_densenet201(base_dir='/home/fi5666wi/Python/PRIAS'):
    model_path = os.path.join(base_dir, 'supervised', 'saved_models',
                             'densenet201_2023-06-16', 'version_0', 'last.pth')

    model = densenet201(num_classes=1, use_classifier=True)
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights, strict=True)
    return model, 1920

def load_uni():
    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    return model, transform

def load_uni_v2(token, img_size=280):
    login(token = token, add_to_git_credential=False)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
                'img_size': 280, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()

    return model, transform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default="/home/fi5666wi/PRIAS_data/features_uni_v2_augment")
    parser.add_argument("-IN", "--use_imagenet", type=bool, default=False)
    parser.add_argument("-UNI", "--use_uni", type=bool, default=True)
    parser.add_argument("--xcl_file", type=str, default="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx")
    parser.add_argument("-x", "--xcl_num", type=int, default=0, help="1 for train, 2 for test, 0 for all")
    parser.add_argument("-r", "--resnet", action='store_true')
    parser.add_argument("--num_features", type=int, default=-1, help="Number of patches to extract from WSI, originally 256, Select -1 for all")  # originally 256
    parser.add_argument("-c", "--create_stitches", type=bool, default=False)
    parser.add_argument("--csv", type=bool, default=True, help="Write to csv file")
    parser.add_argument("--fe_level", default=-6, choices=[-6, -2, -1], help="Which level in the network to extract features from, -6 for densenet lower, -2 for imagenet")
    parser.add_argument("--overlap", type=int, default=75, help="Patch overlap, 75 for 25% overlap, 29 for 10% overlap")
    parser.add_argument("--patch_sz", type=int, default=280, help="Patch size for feature extraction, eg. 299 for GG, 288 for UNI, 280 for UNI2")
    return parser.parse_args()



def write_stuff_to_excel(res_dict):
    import pandas as pd
    import itertools

    flattened_dict = {}
    for key, val in res_dict.items():
        if all(isinstance(i, list) for i in val):
            # val is list of lists
            flat = list(itertools.chain.from_iterable(val))
        else:
            # val is already flat or mixed — just flatten 1 level
            flat = list(val)
        flattened_dict[key] = flat

    max_len = max(len(v) for v in flattened_dict.values())

    for key, val in flattened_dict.items():
        flattened_dict[key] = val + [None] * (max_len - len(val))

    df = pd.DataFrame(flattened_dict)
    df.to_excel("/home/fi5666wi/Python/PRIAS/res_dict_output.xlsx", index=False)


if __name__ == "__main__":
    """
    Main function for extracting features from WSIs
    Loads wsis for each patient and extracts features from the patches with the highest proboability of cancer
    Can choose between a ResNet50, DenseNet201 or DenseNet201 pretrained on ImageNet
    Uses PatientFeature object to extract features and save in .h5 files for each patient
     - If args.csv is True, writes a .csv file with the wsi names and number of patches used in each
     - If args.create_stitches is True, creates a stitched image of the patches used for each wsi (downsampled)
    """
    args = parse_args()
    wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    data = PRIAS_Data_Object(sheet_path, sheet_name=0)

    if args.num_features < 0:
        save_path = f"{args.save_path}_all"
        args.num_features = np.inf
    else:
        save_path = f"{args.save_path}_{args.num_features}" #os.path.join(pat_path, 'patches')


    transform = None
    if args.use_imagenet:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)  
    elif args.use_uni:
        model, transform = load_uni_v2()
    else:
        model, _ = load_densenet201()

    xcl_path = args.xcl_file
    list_of_patients = pd.read_excel(xcl_path, sheet_name=args.xcl_num)['Patient number']

    res_dict = {'no_features': [], 'no_h5s': [], 'no_wsi': [], 'pids': []}
    for pid in list_of_patients:
        print(f"----Patient {pid}----")
        wsi_dict = data.get_patient_data(patient_no=pid)

        patient = PatientFeaturesAugment(pid=pid, base_dir=wsi_dir, wsi_dict=wsi_dict, model=model, transform=transform,
                                  num_patches=args.num_features, patch_overlap=args.overlap, fe_level=args.fe_level,
                                  save_path=save_path, last_visit=False, fe_patch_sz=args.patch_sz, use_uni=args.use_uni)
        #patient.select_and_save(n_iters=10)
        patient.check_coords()
        
        """no_h5s, no_feats, no_wsi = patient.have_a_look()
        if len(no_h5s) > 0:
            res_dict['no_h5s'].append(no_h5s)
            res_dict['pids'].append(pid)
        if len(no_feats) > 0:
            res_dict['no_features'].append(no_feats)
            res_dict['pids'].append(pid)
        if len(no_wsi) > 0:
            res_dict['no_wsi'].append(no_wsi)
            res_dict['pids'].append(pid)"""

    #print(res_dict)
    #write_stuff_to_excel(res_dict)

    print('herå')





