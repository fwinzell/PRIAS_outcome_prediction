import torch
import cv2
import argparse
import os
from torch import nn
import torch.nn.functional as F

from torchvision import models
from torchmetrics import ConfusionMatrix, ROC, AUROC
from prettytable import PrettyTable
from tqdm import tqdm

from wsi_loader import WSIDataset
from torch.utils.data import DataLoader

#from PRIAS.supervised.test import get_cnn
from PRIAS.prias_file_reader import PRIAS_Data_Object
from inference import make_prediction, parse_args, print_pretty_table, get_fancy_confusion_matrix, area_results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
import random
import re

from sklearn.metrics import auc

def classify_with_maps(wsi_dict, diag_dict, label, thr, verbose):
    preds = []
    for j, wsi in enumerate(wsi_dict.keys()):
        if label == 1 and j != len(wsi_dict.keys())-1:
            # If label is 1 then only the last visit is used, so skip to it
            continue

        #wsi_name = wsi_path.split('/')[-1].split('.')[0]
        
        # Load the segmentation
        for num in wsi_dict[wsi]:
            seg = np.load(os.path.join(args.save_path, "maps", f"{wsi}_{num}_10x_segmentation.npy"))
            areas = area_results(seg)

        if len(areas) == 0:
            continue

        y_pred = make_prediction(areas, diag_dict[j][0], diag_dict[j][1], j, thr=thr)

        if verbose:
            print(f"WSI: {wsi}: Label: {label} Predicted: {y_pred}")
        preds.append(y_pred)

    return preds


def load_and_classify(path, wsi_dict, diag_dict,label,thr,verbose):
    res_df = pd.read_csv(path)

    prev_gg, prev_pos = 0, 0
    preds = []
    for j, wsi in enumerate(wsi_dict.keys()):

        if label == 1 and j != len(wsi_dict.keys())-1:
            # If label is 1 then only the last visit is used, so skip to it
            continue

        if j > 0:
            (prev_gg, prev_pos) = diag_dict[j]

        filt = res_df[res_df['WSI'].str.contains(wsi)]
        cancer_perc = np.array([tuple(map(float, re.findall(r'\d*\.\d+|\d+', x))) for x in filt['Cancer'].tolist()]) 

        if len(cancer_perc) == 0:
            continue

        y_pred = make_prediction(cancer_perc, prev_gg, prev_pos, j, thr=thr)

        if verbose:
            print(f"WSI: {wsi}: Label: {label} Predicted: {y_pred}")
        preds.append(y_pred)

    return preds

def calculate_metrics(y_pred, y_true, verbose=False):
    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = torch.logical_and(y_pred == 1, y_true == 1).sum().item()
    FP = torch.logical_and(y_pred == 1, y_true == 0).sum().item()
    TN = torch.logical_and(y_pred == 0, y_true == 0).sum().item()
    FN = torch.logical_and(y_pred == 0, y_true == 1).sum().item()

    # Calculate metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Balanced Accuracy: {balanced_accuracy}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")

        print(f"Confusion Matrix:")
        confmat = ConfusionMatrix("binary", num_classes=2)
        conf = confmat(y_pred, y_true)
        conf = conf.cpu().numpy()
        print_pretty_table(conf, ["active", "treated"])

    return accuracy, balanced_accuracy, sensitivity, specificity

def predict(res_path, list_of_patients, labels, data, thr, verbose=False):
    PRED = []
    TRUE = []
    for i, p in enumerate(list_of_patients):
        if verbose:
            print(f"****Patient {p}****")
        wsi_dict = data.get_patient_data(patient_no=p)
        diag_dict = data.get_gleason_diagnosis(patient_no=p)
        pred = load_and_classify(res_path, wsi_dict, diag_dict, labels[i], thr, verbose=verbose)
        PRED += pred
        TRUE += [labels[i] for _ in range(len(pred))]


    y_pred, y_true = torch.tensor(PRED), torch.tensor(TRUE)
    return calculate_metrics(y_pred, y_true, verbose=False)

def run_iteration(res_path, list_of_patients, labels, data):
    thresholds = np.arange(0, 1200000, 40000)
    sensitivity = []
    specificity = []
    for thr in tqdm(thresholds):
        acc, bac, sens, spec = predict(res_path, list_of_patients, labels, data, thr, verbose=False)
        sensitivity.append(sens)
        specificity.append(spec)

    tpr = np.array(sensitivity)
    fpr = 1 - np.array(specificity)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")

    # Plot ROC curve
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

    return roc_auc, bac, acc

def fprtpr(res_path, list_of_patients, labels, data):
    thresholds = np.arange(0, 1200000, 40000)
    sensitivity = []
    specificity = []
    for thr in tqdm(thresholds):
        _, _, sens, spec = predict(res_path, list_of_patients, labels, data, thr, verbose=False)
        sensitivity.append(sens)
        specificity.append(spec)

    tpr = np.array(sensitivity)
    fpr = 1 - np.array(specificity)

    return fpr, tpr


def bootstrap_for_R(p_df, res_path, data, n_bootstraps=1000):
    all_fpr, all_tpr = fprtpr(res_path, p_df['list_of_patients'].tolist(), p_df['labels'].tolist(), data)

    TPRs = np.zeros((len(all_fpr), n_bootstraps+1))
    TPRs[:, 0] = all_tpr
    for i in range(n_bootstraps):
        sample_df = p_df.sample(frac=1, replace=True)
        fpr, tpr = fprtpr(res_path, sample_df['list_of_patients'].tolist(), sample_df['labels'].tolist(), data)
        tpr_hat = np.interp(all_fpr, fpr[::-1], tpr[::-1])
        TPRs[:, i+1] = tpr_hat

    # save to .csv
    res = np.concatenate((np.expand_dims(all_fpr, 1), TPRs), axis=1)
    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(args.save_path, "fprtpr_for_R3.csv"))


def bootstrap_for_py(p_df, res_path, data, n_bootstraps=1000):
    AUCs = np.zeros(n_bootstraps)
    BAcs = np.zeros(n_bootstraps)
    Accs = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        sample_df = p_df.sample(frac=1, replace=True)
        list_of_patients = sample_df['list_of_patients'].tolist()
        labels = sample_df['labels'].tolist()
        print(f"Bootstrap: {i} ratio: {np.sum(labels)/len(labels)}")
        roc_auc, bac, acc = run_iteration(res_path, list_of_patients, labels, data)
        AUCs[i] = roc_auc
        BAcs[i] = bac
        Accs[i] = acc
    
    print(f"Mean AUC: {np.mean(AUCs)} ({np.std(AUCs)})")
    print(f"Mean ACC: {np.mean(Accs)} ({np.std(Accs)})")
    print(f"Mean BAC: {np.mean(BAcs)} ({np.std(BAcs)})")

    return AUCs, BAcs, Accs


if __name__ == '__main__':
    save_to_csv = True
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    model_path = "/home/fi5666wi/Python/PRIAS/supervised/saved_models/densenet201_2024-01-26/version_1/best.pth"
    data = PRIAS_Data_Object(sheet_path, sheet_name=0)

    args = parse_args()
    xcl_path = args.xcl_file
    p_df = pd.DataFrame({
        "list_of_patients": pd.read_excel(xcl_path, sheet_name=args.xcl_num)['Patient number'],
        "labels": pd.read_excel(xcl_path, sheet_name=args.xcl_num)['act0 treated1']
    })
    res_path = os.path.join(args.save_path, "results_gleason_grading_51.csv")

    #bootstrap_for_R(p_df, res_path, data, n_bootstraps=1000)
    aucs, bacs, accs = bootstrap_for_py(p_df, res_path, data, n_bootstraps=1000)

    if save_to_csv:
        boot_df = pd.DataFrame({'AUC': aucs, 'Acc': accs, 'BAc': bacs})
        fname = f"/home/fi5666wi/R/PRIAS/Bootstrap_protocol"
        csv_path = fname + ".csv"
        i = 1
        while os.path.exists(csv_path):
            csv_path = fname + f"_{i}.csv"
            i += 1
        boot_df.to_csv(csv_path, index=False)

    plt.show()



    



    
    