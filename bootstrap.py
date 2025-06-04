import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix, ROC, AUROC
from prettytable import PrettyTable
import seaborn as sn

from models import PRIAS_Model
from dataset import  PRIAS_Feature_Dataset, PRIAS_Generic_Dataset, CrossValidation_Dataset
from prias_file_reader import PRIAS_Data_Object
from cross_validation import parse_config
#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from eval import plot_binary, get_eval_dataset, load_model
import random
import yaml

def run(model, loader, device):
    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))
    pids = torch.zeros(len(loader))
    with torch.no_grad():
        for i, (pid, feats, label) in enumerate(loader):
            feats, label = feats.to(device), label.to(device)
            _, y_prob, y_hat = model(feats.squeeze())
            probs[i] = y_prob
            preds[i] = y_hat
            true[i] = label
            pids[i] = pid

    return pd.DataFrame({'probs': probs, 'preds': preds, 'true': true, 'pids': pids})

    

def boostrap(model, device, base_dir, verbose=True, n_bootstraps=1000):
    model = model.to(device)
    dataset = get_eval_dataset(base_dir, xls_sheet_name=2, use_last_visit=False, use_features=True)

    loader = tqdm(DataLoader(dataset, batch_size=1))
    res_df= run(model, loader, device)

    aucs = []
    accs = []
    bacs = []
    dasboot = {}
    for i in range(n_bootstraps):
        print(f"Bootstrap: {i}")
        # Sample with replacement
        sampled_df = res_df.sample(n=len(res_df), replace=True)

        y_preds = torch.tensor(sampled_df['preds'].values)
        y_probs = torch.tensor(sampled_df['probs'].values)
        y_true = torch.tensor(sampled_df['true'].values)

        auc, acc, bac = plot_binary(y_preds, 
                               y_probs, 
                               y_true, 
                               plot=False)
        aucs.append(auc)
        accs.append(acc)
        bacs.append(bac)
        dasboot[f"probs_{i}"] = y_probs
        dasboot[f"true_{i}"] = y_true

    print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
    print(f"Mean ACC: {np.mean(accs)} ({np.std(accs)})")
    print(f"Mean BAC: {np.mean(bacs)} ({np.std(bacs)})")

    return aucs, dasboot


def boostrap_patient_level(model, device, base_dir, verbose=True, n_bootstraps=1000):
    model = model.to(device)
    dataset = get_eval_dataset(base_dir, xls_sheet_name=2, use_last_visit=False, use_features=True)

    loader = tqdm(DataLoader(dataset, batch_size=1))
    res_df= run(model, loader, device)

    patients = res_df['pids'].unique().astype(int)
    aucs = []
    accs = []
    bacs = []
    dasboot = {}
    for i in range(n_bootstraps):
        print(f"Bootstrap: {i}")
        # Sample with replacement
        sampled_patients = random.choices(patients, k=len(patients))
        sampled_df = res_df[res_df['pids'].isin(sampled_patients)]

        y_preds = torch.tensor(sampled_df['preds'].values)
        y_probs = torch.tensor(sampled_df['probs'].values)
        y_true = torch.tensor(sampled_df['true'].values)

        auc, acc, bac = plot_binary(y_preds, 
                               y_probs, 
                               y_true, 
                               plot=False)
        aucs.append(auc.item())
        accs.append(acc.item())
        bacs.append(bac.item())
        dasboot[f"probs_{i}"] = y_probs
        dasboot[f"true_{i}"] = y_true

    print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
    print(f"Mean ACC: {np.mean(accs)} ({np.std(accs)})")
    print(f"Mean BAC: {np.mean(bacs)} ({np.std(bacs)})")

    return aucs, accs, bacs, dasboot


def bootstrap_offline(csv_path, n_bootstraps=1000):
    df = pd.read_csv(csv_path)
    aucs = []
    accs = []
    bacs = []
    dasboot = {}
    
    for i in range(n_bootstraps):
        print(f"Bootstrap: {i}")
        # Sample with replacement
        sampled_df = df.sample(n=len(df), replace=True)

        y_preds = torch.tensor(sampled_df['preds'].values)
        y_probs = torch.tensor(sampled_df['probs'].values)
        y_true = torch.tensor(sampled_df['true'].values)

        auc, acc, bac = plot_binary(y_preds, 
                               y_probs, 
                               y_true, 
                               plot=False)
        aucs.append(auc.item())
        accs.append(acc.item())
        bacs.append(bac.item())
        dasboot[f"probs_{i}"] = y_probs
        dasboot[f"true_{i}"] = y_true

    print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
    print(f"Mean ACC: {np.mean(accs)} ({np.std(accs)})")
    print(f"Mean BAC: {np.mean(bacs)} ({np.std(bacs)})")

    return aucs, accs, bacs, dasboot




if __name__ == "__main__":
    save_to_csv = True
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #model_path = os.path.join(config.save_dir, "imagenet201_gated_2023-07-31", "models", "version_1", "last.pth")
                            # 'cross_val_2023-06-26/run_1/models', 'version_0',
                            # 'densenet201_gated_fold_0last.pth')

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models",
                             "densenet_gated_2024-12-09", "models", "version_0")

    #"vitL_gated_2024-12-09", "models", "version_0")
    #"imagenet_gated_2024-12-09", "models", "version_0")
    #"densenet_gated_2024-12-09", "models", "version_0")

    fyaml = os.path.join(model_dir, "config.yaml")
    if os.path.exists(fyaml):
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    model_path = os.path.join(model_dir, "best.pth")
    model = load_model(config, model_path, config.num_features, long_mode=False)

    #aucs, boot = boostrap(model, device, config.base_dir, verbose=False, n_bootstraps=1000)
    aucs, accs, bacs, _ = boostrap_patient_level(model, device, config.base_dir, verbose=False, n_bootstraps=1000)"""

    model_names = ["cv_vitH_basic", "cv_vitH_basic_old", "cv_vitH_aem", "cv_vitH_sks", "cv_vitH_gg"]
    for model_name in model_names:
        aucs, accs, bacs, boot = bootstrap_offline(f"/home/fi5666wi/R/PRIAS/Res_{model_name}.csv", n_bootstraps=1000)
        if save_to_csv:
            boot_df = pd.DataFrame({'AUC': aucs, 'Acc': accs, 'BAc': bacs})
            fname = f"/home/fi5666wi/R/PRIAS/Bootstrap_{model_name}"
            csv_path = fname + ".csv"
            i = 1
            while os.path.exists(csv_path):
                csv_path = fname + f"_{i}.csv"
                i += 1
            boot_df.to_csv(csv_path, index=False)




   




