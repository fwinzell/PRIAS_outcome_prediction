import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from PRIAS.protocol.wsi_loader import WSIDataset
import random

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

from PRIAS.prias_file_reader import PRIAS_Data_Object
from PRIAS.supervised.resnet import resnet50
from PRIAS.supervised.densenet import densenet201
from PRIAS.extract_patches import PatientObject
from PRIAS.extract_features import load_densenet201

#UNI
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

from UNI.uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

PIL.Image.MAX_IMAGE_PIXELS = None


class PatientFeatures2(PatientObject):
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
                 fe_level, save_path, last_visit=True, fe_patch_sz=299, use_uni=False, add_gg_scores=False, gg_dir=None):
        super(PatientFeatures2, self).__init__(pid,
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
        self.add_gg_scores = add_gg_scores
        
        # feature extractor model
        self.fe_model = model.to(self.device)
        if self.add_gg_scores:
            self.gg_net, _ = load_densenet201(gg_dir)
            self.gg_net = self.gg_net.to(self.device)

        self.transform = transform
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
        dataset = WSIDataset(wsi_path, patch_size=self.patch_sz, overlap=self.overlap)
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
        dataset = WSIDataset(wsi_path, patch_size=self.fe_patch_sz, overlap=self.overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        feature_dict = extract_patch_features_from_dataloader(self.fe_model, dataloader)

        # convert these to torch
        features = torch.Tensor(feature_dict['embeddings'])
        pts = torch.Tensor(feature_dict['labels']).type(torch.long)

        return features, pts

    def _get_gg_scores(self, wsi_path, batch_size=10):
        dataset = WSIDataset(wsi_path, patch_size=self.fe_patch_sz, overlap=self.overlap)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        pts = torch.zeros((len(dataset), 2), device=self.device)
        probs = torch.zeros(len(dataset), device=self.device)
        for i, (data, coords) in enumerate(dataloader):
            data, coords = data.to(self.device), coords.to(self.device)

            with torch.no_grad():
                y_hat = self.gg_net(data)
                probs[i*batch_size:(i+1)*batch_size] = y_hat.squeeze()
                pts[i*batch_size:(i+1)*batch_size, :] = coords

        return probs.cpu(), pts.int().cpu()

    def select_and_save(self):
        """
        Selects the n patches with the highest probability of cancer for each wsi
        Saves the features, coordinates and attention scores in a .h5 file from the n patches of each wsi
        Uses init_hdf5() to initialize the .h5 file
        Uses _get_wsi_scores_and_features() to extract features and probabilities
        Uses _save_to_hdf5() to save features, coordinates and probabilities to .h5 file
        Returns nothing
        """
        for wsi_id in self.wsi_dict.keys():
            print(f'Processing: {wsi_id}')
            if os.path.exists(os.path.join(save_path, wsi_id)+'.h5'):
                print("Already exists, skipping...")
                # Remove wsi from data frame (useful when writng to .csv file)
                rm = [self.wsi_list.index(wsi) for wsi in self.wsi_list if wsi[:10] == wsi_id]
                self.df = self.df.drop(rm)
                continue
            all_pts = []
            all_features = []
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

            n_samples = min(len(all_features), self.n)
            if len(all_features) < self.n:
                print(f"Warning: Less than {self.n} patches for {wsi_id}")

            indxs = random.sample(range(len(all_features)), n_samples)

            for ii in indxs:
                pt = all_pts[ii]
                features = all_features[ii]
                #print(f"Non-zero: {torch.nonzero(features)}")

                if self.hdf5_file is None:
                    self.hdf5_file = init_hdf5(features, pt, self.save_path, self.pid, wsi_id, save_coord=True)
                else:
                    self._save_to_hdf5(features, pt)


                self.df.loc[self.wsi_list.index(f"{wsi_id}-{pt[0]}"), 'n_patches'] += 1

            self.patch_indxs = {wsi_id: indxs}
            self.pts = {wsi_id: all_pts}
            self.hdf5_file = None
            print('Done')

        if self.add_gg_scores:
            self.add_gg_scores_to_h5()

    def add_gg_scores_to_h5(self):
        for wsi_id in self.wsi_dict.keys():
            h5_fname = os.path.join(self.save_path, wsi_id)+'.h5'
            if os.path.exists(h5_fname):
                with h5py.File(h5_fname, 'a') as file:
                    if 'gg_scores' in file:
                        gg_scores = file['gg_scores'][:]
                        if len(gg_scores) != len(file['coords'][:]):
                            print(f"GG scores length mismatch for {wsi_id}, deleting...")
                            del file['gg_scores']
                        else:
                            print(f"GG scores already exist for {wsi_id}, skipping...")
                            continue
                    
                    print(f"Adding GG scores to {wsi_id}")
                    all_pts = []
                    all_probs = []
                    for num in self.wsi_dict[wsi_id]:
                        wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")
                        if os.path.exists(wsi_path):
                            y_probs, pts = self._get_gg_scores(wsi_path, batch_size=100)
                            all_probs = all_probs + y_probs.tolist()
                            wsi_ind = torch.ones((pts.shape[0], 1))*num
                            pts_list = torch.cat((wsi_ind.int(), pts), dim=1)
                            all_pts = all_pts + pts_list.tolist()

                    coords = file['coords'][:]
                    # Arrange all_pts to match the coords in the file
                    row_to_index = {tuple(row): i for i, row in enumerate(all_pts)}
                    order = [row_to_index[tuple(row)] for row in coords]
                    all_probs = np.array(all_probs)[order]
                    file.create_dataset('gg_scores', data=all_probs, dtype=np.float32)
                    print(f"GG scores added to {wsi_id}")
                    
        print("Donzoe")
                        

    def _save_to_hdf5(self, features, pt):
        """
        Save features and coordinates to respective dataset of initialized .h5 file
        """
        f_vec = np.array(features)[np.newaxis, ...]
        shape = f_vec.shape

        file = h5py.File(self.hdf5_file, 'a')
        dset = file['features']
        dset.resize(len(dset) + shape[0], axis=0)
        dset[-shape[0]:] = f_vec

        if 'coords' in file:
            coord_dset = file['coords']
            coord_dset.resize(len(coord_dset) + shape[0], axis=0)
            coord_dset[-shape[0]:] = pt

        file.close()

def init_hdf5(features, pt, save_path, pid, wsi_id, save_coord=False):
    """
    Initialize a .h5 file for saving features, coordinates and probabilities
    Documentations: https://docs.h5py.org/en/stable/high/file.html https://docs.h5py.org/en/stable/high/dataset.html
    Attributes:
        patient: patient id
    Datasets:
        features
        coords: coordinates of patch (top left corner)
    """
    file_path = os.path.join(save_path, wsi_id)+'.h5'
    file = h5py.File(file_path, "w")
    file.attrs['patient'] = pid
    f_vec = np.array(features)[np.newaxis,...]
    dtype = f_vec.dtype

    # Initialize a resizable dataset to hold the output
    shape = f_vec.shape
    maxshape = (None,) + shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
    dset = file.create_dataset('features',
                                shape=shape, maxshape=maxshape,  chunks=shape, dtype=dtype)

    dset[:] = f_vec

    if save_coord:
        coord_dset = file.create_dataset('coords', shape=(1, 3), maxshape=(None, 3), chunks=(1, 3), dtype=np.int32)
        coord_dset[:] = pt

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

def load_uni_v2(token):
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
    parser.add_argument("-s", "--save_path", type=str, default="/home/fi5666wi/PRIAS_data/features_uni_v2")
    parser.add_argument("-IN", "--use_imagenet", type=bool, default=False)
    parser.add_argument("-UNI", "--use_uni", type=bool, default=True)
    parser.add_argument("--xcl_file", type=str, default="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx")
    parser.add_argument("-x", "--xcl_num", type=int, default=0, help="1 for train, 2 for test, 0 for all")
    parser.add_argument("-r", "--resnet", action='store_true')
    parser.add_argument("--num_features", type=int, default=-1, help="Number of patches to extract from WSI, originally 256, Select -1 for all")  # originally 256
    parser.add_argument("-c", "--create_stitches", type=bool, default=False)
    parser.add_argument("--csv", type=bool, default=False, help="Write to csv file")
    parser.add_argument("--fe_level", default=-6, choices=[-6, -2, -1], help="Which level in the network to extract features from, -6 for densenet lower, -2 for imagenet")
    parser.add_argument("--overlap", type=int, default=75, help="Patch overlap, 75 for 25% overlap, 29 for 10% overlap")
    parser.add_argument("--patch_sz", type=int, default=280, help="Patch size for feature extraction, eg. 299 for GG, 288 for UNI, 280 for UNI2")
    parser.add_argument("--add_gg_scores", type=bool, default=True, help="Add GG scores to the features")
    parser.add_argument("--prias_sheet_dir", type=str, default="/home/fi5666wi/Documents/PRIAS sheets")
    parser.add_argument("--wsi_dir", type=str, default="/home/fi5666wi/PRIAS_data/wsis")
    parser.add_argument("-b", "--base_dir", type=str, default="/home/fi5666wi/Python/PRIAS")
    return parser.parse_args()


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
    wsi_dir = args.wsi_dir
    sheet_path = f"{args.prias_sheet_dir}/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
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

    for pid in list_of_patients:
        print(f"----Patient {pid}----")
        wsi_dict = data.get_patient_data(patient_no=pid)

        patient = PatientFeatures2(pid=pid, base_dir=wsi_dir, wsi_dict=wsi_dict, model=model, transform=transform,
                                  num_patches=args.num_features, patch_overlap=args.overlap, fe_level=args.fe_level,
                                  save_path=save_path, last_visit=False, fe_patch_sz=args.patch_sz, use_uni=args.use_uni, 
                                  add_gg_scores=args.add_gg_scores, gg_dir = args.base_dir)
        #patient.select_and_save()
        patient.add_gg_scores_to_h5()
        if args.csv:
            patient.write_df_to_csv(os.path.join(save_path, 'wsi_list.csv'))
        if args.create_stitches:
            patient.create_stitches(downsample_factor=0.1, save_dir=os.path.join(save_path, 'stitches'))

    #patient.select_and_save()
    #patient.save_patches()
    #patient.write_df_to_csv(os.path.join(save_path, 'wsi_list.csv'))
    #patient.create_stitches(downsample_factor=0.1, save_dir=os.path.join(save_path, 'stitches'))
    print('herå')





