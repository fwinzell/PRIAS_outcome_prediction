import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from load_slides import SNDSlideDataset
from PRIAS.extract_features import load_uni, load_densenet201
from PRIAS.extract_features_2 import load_uni_v2

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

#UNI
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

from UNI.uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

PIL.Image.MAX_IMAGE_PIXELS = None

class SNDPatientFeatures(object):
    """
    Class for extracting features from the SND slides
    Builds on PatientFeatures class from extract_features.py
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
        magnification_index: magnification level of patches (default 0, 0 = highest magnification)
    """
    def __init__(self, pid, base_dir, model, pt_model, transform, num_patches, patch_overlap, save_path, magnification_index=0, patch_sz=299, gg_scores=False, uni_patch_sz=288):
        self.pid = pid
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        self.gg_model = model.to(self.device)
        self.uni_model = pt_model.to(self.device)

        self.transform = transform
        self.patch_overlap = patch_overlap
        self.save_path = save_path
        self.magnification = magnification_index

        self.n = num_patches
        self.gg_scores = gg_scores
        self.select_all = True if num_patches == np.inf else False
        #if self.select_all:
        #    self.n = np.inf

        if save_path is not None:
            self.save_path = save_path
            os.makedirs(save_path, exist_ok=True)

        self.patch_size = patch_sz
        self.uni_patch_sz = uni_patch_sz
        self.attn_scores = torch.Tensor
        self.patch_indxs = torch.Tensor
        self.pts = []
        self.hdf5_file = None

        self.slide_paths = [f for f in os.listdir(self.base_dir) if f.startswith(pid) and f.endswith('.mrxs')]
        
        generic_out = self.uni_model(torch.zeros((1,3,self.uni_patch_sz,self.uni_patch_sz)).to(self.device))
        self.feature_shape = generic_out.shape[1:]

    def _get_wsi_scores_and_features(self, slide_name, batch_size=10):
        """
        Extract features from a given wsi
        Uses WSIDataset class to split WSI into patches
        Returns:
            features: LxNx1x1 features (L = number of patches in wsi)
            atten_scores: Lx1 attention scores i.e. probability of cancer
            pts: Lx2 coordinates of patches (top left corner)
        """
        # N: 2048 for resnet50
        dataset = SNDSlideDataset(slide_name, patch_size=self.patch_size, overlap=self.patch_overlap, segmentation_level=3, magnification_level=self.magnification)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        atten_scores = torch.zeros(len(dataset), device=self.device)
        pts = torch.zeros((len(dataset), 2), device=self.device)
        features = torch.zeros(tuple([len(dataset), self.feature_shape[0], 1, 1]), device=self.device)
        for i, (data, coords) in enumerate(dataloader):
            data, coords = data.to(self.device), coords.to(self.device)

            with torch.no_grad():
                crop = CenterCrop(self.uni_patch_sz)(data)
                h = self.uni_model(crop)
                y_hat = torch.sigmoid(self.gg_model(data))
                prob = y_hat.squeeze()    
                
                while h.dim() < 4:
                    h = h.unsqueeze(-1)
                atten_scores[i*batch_size:(i+1)*batch_size] = prob
                pts[i*batch_size:(i+1)*batch_size, :] = coords
                features[i*batch_size:(i+1)*batch_size, :] = h

        return features.cpu(), atten_scores.cpu(), pts.int().cpu()
    
    def _get_gg_scores(self, slide_name, batch_size=10):
        """
        Extract GG scores from a given wsi
        Uses WSIDataset class to split WSI into patches
        Returns:
            atten_scores: Lx1 attention scores i.e. probability of cancer
            pts: Lx2 coordinates of patches (top left corner)
        """
        dataset = SNDSlideDataset(slide_name, patch_size=self.patch_size, overlap=self.patch_overlap, segmentation_level=3, magnification_level=self.magnification)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        atten_scores = torch.zeros(len(dataset), device=self.device)
        pts = torch.zeros((len(dataset), 2), device=self.device)
        for i, (data, coords) in enumerate(dataloader):
            data, coords = data.to(self.device), coords.to(self.device)

            with torch.no_grad():
                y_hat = torch.sigmoid(self.gg_model(data))
                prob = y_hat.squeeze()
                atten_scores[i*batch_size:(i+1)*batch_size] = prob
                pts[i*batch_size:(i+1)*batch_size, :] = coords

        return atten_scores.cpu(), pts.int().cpu()
    

    def _get_wsi_features(self, slide_name, batch_size=10):
        """
        Extract all features from a given wsi
        Uses WSIDataset class to split WSI into patches
        Returns:
            features: LxNx1x1 features (L = number of patches in wsi)
            pts: Lx2 coordinates of patches (top left corner)
        """
        # N: 2048 for resnet50
        dataset = SNDSlideDataset(slide_name, patch_size=self.patch_size, overlap=self.patch_overlap, segmentation_level=3, magnification_level=self.magnification)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

        pts = torch.zeros((len(dataset), 2), device=self.device)
        features = torch.zeros(tuple([len(dataset), self.feature_shape[0], 1, 1]), device=self.device)
        for i, (data, coords) in enumerate(dataloader):
            data, coords = data.to(self.device), coords.to(self.device)

            with torch.no_grad():
                h = self.uni_model(data)  
                while h.dim() < 4:
                    h = h.unsqueeze(-1)
                pts[i*batch_size:(i+1)*batch_size, :] = coords
                features[i*batch_size:(i+1)*batch_size, :] = h

        return features.cpu(), pts.int().cpu()
    
    def select_and_save(self):
        """
        Selects the n patches with the highest probability of cancer
        Saves the features, coordinates and attention scores in a .h5 file from the n patches of each biopsy
        Uses init_hdf5() to initialize the .h5 file
        Uses _get_wsi_scores_and_features() to extract features and probabilities
        Uses _save_to_hdf5() to save features, coordinates and probabilities to .h5 file
        Returns nothing
        """

        all_scores = []
        all_pts = []
        all_features = []
        for slide_name in self.slide_paths:
            print(f'Processing: {slide_name}')
            
            if not self.gg_scores:
                f, pts = self._get_wsi_features(slide_name)
            else: 
                f, atten, pts = self._get_wsi_scores_and_features(slide_name)
                all_scores = all_scores + atten.tolist()
            all_features = all_features + f.tolist()
            all_pts = all_pts + pts.tolist()

        if self.select_all:
            indxs = np.arange(len(all_features))
        else:
            scores = torch.Tensor(all_scores)

            attn_scores, order = torch.sort(scores, descending=True)
            indxs = order[:self.n]
            patch_indxs, _ = torch.sort(indxs)

        
        n_patches = min(self.n, len(all_features))
        
        for j in range(n_patches):
            features = all_features[indxs[j]]
            pts = all_pts[indxs[j]]

            if not self.gg_scores:
                if self.hdf5_file is None:
                    self.hdf5_file = self.init_hdf5(features, pts)
                else:
                    self._save_to_hdf5(features, pts)
            else:
                gg_prob = all_scores[indxs[j]]
                if self.hdf5_file is None:
                    self.hdf5_file = self.init_hdf5(features, gg_prob, pts)
                else:
                    self._save_to_hdf5(features, gg_prob, pts)

        if not self.select_all:
            self.attn_scores = attn_scores
            self.patch_indxs = patch_indxs
        print('Done')

    def add_gg_scores(self):
        all_scores = []
        for slide_name in self.slide_paths:
            print(f'Processing: {slide_name}')
            
            ggs, pts = self._get_gg_scores(slide_name)
            all_scores = all_scores + ggs.tolist()

        scores = torch.Tensor(all_scores)
    

    def _save_to_hdf5(self, features, atten=None, pts=None):
        """
        Save features and probabilities to respective dataset of initialized .h5 file
        """
        f_vec = np.array(features)[np.newaxis, ...]
        shape = f_vec.shape

        file = h5py.File(self.hdf5_file, 'a')
        dset = file['features']
        dset.resize(len(dset) + shape[0], axis=0)
        dset[-shape[0]:] = f_vec

        if self.gg_scores:
            atten_dset = file['gg_score']
            atten_dset.resize(len(atten_dset) + shape[0], axis=0)
            atten_dset[-shape[0]:] = atten

        if pts is not None:
            pts = np.array(pts)[np.newaxis, ...]
            pts_dset = file['coords']
            pts_dset.resize(len(pts_dset) + shape[0], axis=0)
            pts_dset[-shape[0]:] = pts

        file.close()

    def init_hdf5(self, features, attn=None, pts=None):
        """
        Initialize a .h5 file for saving features, coordinates and probabilities
        Documentations: https://docs.h5py.org/en/stable/high/file.html https://docs.h5py.org/en/stable/high/dataset.html
        Attributes:
            patient: patient id
        Datasets:
            features
            score: probability of cancer
        """
        file_path = os.path.join(self.save_path, self.pid)+'.h5'
        file = h5py.File(file_path, "w")
        file.attrs['patient'] = self.pid
        f_vec = np.array(features)[np.newaxis,...]
        dtype = f_vec.dtype

        # Initialize a resizable dataset to hold the output
        shape = f_vec.shape
        maxshape = (None,) + shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
        dset = file.create_dataset('features',
                                    shape=shape, maxshape=maxshape,  chunks=shape, dtype=dtype)

        dset[:] = f_vec
        if self.gg_scores:
            a_dset = file.create_dataset('gg_score', shape=(1, 1), maxshape=(None, 1), dtype=float)
            a_dset[:] = attn

        if pts is not None:
            pts = np.array(pts)[np.newaxis, ...]
            pts_dset = file.create_dataset('coords', data=pts, maxshape=(None, 2), dtype='i')
            pts_dset.resize(len(pts_dset) + shape[0], axis=0)
            pts_dset[-shape[0]:] = pts

        file.close()
        return file_path
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default="/home/fi5666wi/PRIAS_data/features_uni_v2_snd_gg")
    parser.add_argument("--magnification", type=int, default=10)
    parser.add_argument("-IN", "--use_imagenet", type=bool, default=False)
    #parser.add_argument("-UNI", "--use_uni", type=bool, default=False)
    parser.add_argument("--num_features", type=int, default=np.inf)  # originally 256
    parser.add_argument("--overlap", type=int, default=75, help="Patch overlap, 75 for 25% overlap, 29 for 10% overlap")
    parser.add_argument("--patch_sz", type=int, default=280, help="Patch size for feature extraction, eg. 299 for GG, 288 for UNI, 280 for UNI2")
    parser.add_argument("--gg", type=bool, default=True, help="If True, extract features with gg scores from all patches in the WSI")
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
    
    pt_model, transform = load_uni_v2()
    model, f_sz = load_densenet201()
    #f_sz = 1024

    magn_idx = [40, 20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125].index(args.magnification)

    csv_path = "/home/fi5666wi/SND_prostate_cancer/patient_documentation.csv"
    list_of_patients = pd.read_csv(csv_path)['patient_n']

    for pid in list_of_patients:
        print(f"----Patient {pid}----")
        h5_file_path = os.path.join(args.save_path, f"{pid}.h5")
        if os.path.exists(h5_file_path):
            print(f"{h5_file_path} already exists")
            continue
        if pid == 'patient_036' or pid == 'patient_049':
            continue

        patient = SNDPatientFeatures(pid, "/home/fi5666wi/SND_prostate_cancer/", model, pt_model, transform, args.num_features, args.overlap, args.save_path, magnification_index=magn_idx, 
                                     patch_sz=args.patch_sz, gg_scores=args.gg, uni_patch_sz=280)
        patient.select_and_save()

    #patient.select_and_save()
    #patient.save_patches()
    #patient.write_df_to_csv(os.path.join(save_path, 'wsi_list.csv'))
    #patient.create_stitches(downsample_factor=0.1, save_dir=os.path.join(save_path, 'stitches'))
    print('herå')
