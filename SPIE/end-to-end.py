import pandas as pd
import torch
import torchvision.models as models
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt
from scipy import stats

from extract_features import PatientFeatures, load_densenet201, load_uni
from eval import plot_binary, load_model
from dataset import  PRIAS_Feature_Dataset, PRIAS_Generic_Dataset
from prias_file_reader import PRIAS_Data_Object

from torchmetrics import ConfusionMatrix, ROC, AUROC
from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable
import seaborn as sn
import yaml
from time import process_time
from datetime import timedelta


class FeatureExtractor(PatientFeatures):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_features(self, wsi_id):
        """
        Selects the n patches with the highest probability of cancer for each wsi of the biopsy with wsi_id
        Returns the features and attention scores 
        Uses _get_wsi_scores_and_features() to extract features and probabilities
        """
        #for wsi_id in self.wsi_dict.keys():
        #    print(f'Processing: {wsi_id}')
            
        all_scores = []
        #all_pts = []
        all_features = []
        biopsy_missing = True
        for num in self.wsi_dict[wsi_id]:
            wsi_path = os.path.join(self.base_dir, f"{wsi_id}-{num}_10x.png")
            if os.path.exists(wsi_path):
                print(f"WSI: {wsi_id}-{num}")
                biopsy_missing = False
                f, atten, pts = self._get_wsi_scores_and_features(wsi_path)
                all_scores = all_scores + atten.tolist()
                all_features.append(f)

        if biopsy_missing:
            print(f"Biopsy {wsi_id} missing")
            return None, None
            
        scores = torch.Tensor(all_scores)

        attn_scores, order = torch.sort(scores, descending=True)
        indxs = order[:self.n]
        patch_indxs, _ = torch.sort(indxs)
        
        features = torch.cat(all_features, dim=0)
        features = features[patch_indxs]

        return features, scores[indxs]


def get_prias_model(model_dir):
    fyaml = os.path.join(model_dir, "config.yaml")
    if os.path.exists(fyaml):
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    model_path = os.path.join(model_dir, "last.pth")
    model = load_model(config, model_path, config.num_features, long_mode=False)

    return model, config


def outcome_prediction(model, features):
    """
    Predicts the outcome probability for the selected patches
    """
    pids = [int(p) for p in dataset.data_dict['patient_number']]
    slides = [str(s) for s in dataset.data_dict['slides']]
    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))
    with torch.no_grad():
        for i, (pid, feats, label) in enumerate(loader):
            feats, label = feats.to(device), label.to(device)
            logits, y_prob, y_hat = model(feats.squeeze())
            probs[i] = y_prob
            preds[i] = y_hat
            true[i] = label

            print(f"Patient: {pid.item()} \nProbability: {y_prob.item()} Predicition: {y_hat.item()} Label: {label.item()}")
            
    acc = torch.sum(torch.eq(preds, true))/len(loader)

    print(f"Accuracy: {acc}")

     
    return pids, slides, probs, preds, true


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-s", "--save_path", type=str, default="/home/fi5666wi/PRIAS_data/features_imagenet")
    parser.add_argument("--xcl_file", type=str, default="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx")
    parser.add_argument("-x", "--xcl_num", type=int, default=2, help="1 for train, 2 for test, 0 for all")
    #parser.add_argument("-r", "--resnet", action='store_true')
    parser.add_argument("--num_features", type=int, default=256)  # originally 256
    #parser.add_argument("-c", "--create_stitches", type=bool, default=False)
    #parser.add_argument("--csv", type=bool, default=True, help="Write to csv file")
    parser.add_argument("--fe_level", default=-2, choices=[-6, -2, -1], help="Which level in the network to extract features from, -6 for densenet lower, -2 for imagenet")
    parser.add_argument("--overlap", type=int, default=75, help="Patch overlap, 75 for 25% overlap, 29 for 10% overlap")
    #parser.add_argument("--patch_sz", type=int, default=299, help="Patch size for feature extraction, eg. 299 for GG, 288 for UNI")
    return parser.parse_args()


def main(prias_model, use_uni=False, use_imagenet=False):
    """
    Main function for extracting features from WSIs and predicting outcome probability
    Loads wsis for each patient and extracts features from the patches with the highest proboability of cancer
    Can choose between UNI,  GG DenseNet201 or DenseNet201 pretrained on ImageNet
    """
    args = parse_args()
    wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    data = PRIAS_Data_Object(sheet_path, sheet_name=0)

    if use_uni:
        patch_sz = 288
    else:
        patch_sz = 299

    #save_path = args.save_path #os.path.join(pat_path, 'patches')

    # load GG-Net
    model, f_sz = load_densenet201()

    # Load pretrained model (optional)
    transform = None
    if use_imagenet:
        #pt_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        pt_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    elif use_uni:
        pt_model, transform = load_uni()
        f_sz = 1024
    else:
        pt_model = None

    xcl_path = args.xcl_file
    list_of_patients = pd.read_excel(xcl_path, sheet_name=args.xcl_num)['Patient number']

    probs = []
    preds = []
    true = []

    t_slides = []
    t_start = process_time()

    for pid in list_of_patients:
        print(f"----Patient {pid}----")
        wsi_dict = data.get_patient_data(patient_no=pid)
        label = data.get_patient_label(patient_no=pid)

        fe = FeatureExtractor(pid=pid, base_dir=wsi_dir, wsi_dict=wsi_dict, model=model, pt_model=pt_model, transform=transform, 
                            num_patches=args.num_features, patch_overlap=args.overlap, fe_level=args.fe_level,
                            save_path=None, last_visit=False, fe_patch_sz=patch_sz, use_uni=use_uni)

        for i, wsi_id in enumerate(wsi_dict.keys()):  
            t_slide_start = process_time()
            if label == 1 and i+1 < len(wsi_dict): # If label is 1 (treated), check only last case
                continue
            features, scores = fe.get_features(wsi_id)
            if features is None:
                continue
            with torch.no_grad():
                #features, label = features.to(device), label.to(device)
                logits, y_prob, y_hat = prias_model(features.squeeze())
                probs.append(y_prob)
                preds.append(y_hat)
                true.append(label)

                print(f"Patient: {pid} \nProbability: {y_prob.item()} Predicition: {y_hat.item()} Label: {label}")
            t_slide_end = process_time()
            t_slide_elapsed = t_slide_end - t_slide_start
            t_slides.append(t_slide_elapsed)
            print(f"Time for {wsi_id}: {np.round(t_slide_elapsed, 1)}s")


    t_end = process_time()
    t_elapsed = np.round(t_end - t_start)
    acc = torch.sum(torch.eq(torch.Tensor(preds), torch.Tensor(true)))/len(true)
    print(f"Accuracy: {acc}")

    print(f"Total Elapsed Time: {timedelta(seconds = t_elapsed)}")
    print(f"Average Time per Patient: {t_elapsed/len(list_of_patients)}s")
    print(f"Average Time per Biopsy: {t_elapsed/len(true)}s")

    t_slides = np.array(t_slides)
    print(f"Slide times: {np.mean(t_slides)} ({np.std(t_slides)})")
    print(f"95% CI: {stats.t.interval(0.95, len(t_slides)-1, loc=np.mean(t_slides), scale=stats.sem(t_slides))}")
    
    print('herå')



if __name__ == "__main__":
    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models",
                             "vitL_gated_2024-12-09", "models", "version_1")
                             
    #
    #"vitL_gated_2024-07-01", "models", "version_0")
    #'cross_val_2024-01-15/run_2/models', "version_0",
    #"densenet201_gated_2024-07-01", "models", "version_0")

    model, config = get_prias_model(model_dir)
    main(model, use_uni=False, use_imagenet=True)