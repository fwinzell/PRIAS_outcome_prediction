"""
Survival analysis for external dataset
"""

import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import yaml
import h5py

from PRIAS.models import PRIAS_Model
from PRIAS.eval import load_model
from PRIAS.train import parse_config

from eval import SND_Feature_Dataset


def eval_surv(model, device, base_dir, csv_path="/home/fi5666wi/SND_prostate_cancer/patient_documentation.csv", verbose=True):
    pids = [os.path.splitext(f)[0] for f in os.listdir(base_dir)]
    dataset = SND_Feature_Dataset(base_dir, csv_path, pids)
    loader = tqdm(DataLoader(dataset, batch_size=1))

    events = []
    t_preds = []
    pid_list = []

    if verbose:
        plt.figure()

    with torch.no_grad():
        for i, (pid, feats, label, _, _) in enumerate(loader):
            feats = feats.to(device)
            t_pred, *_ = model(feats.squeeze())
            events.append(label.item())
            t_preds.append(t_pred.item())
            pid_list.append(pid)
            if verbose:
                print(f"Patient: {pid} | T (pred): {t_pred.item():.2f} | Event: {label.item()}")

    c_index = concordance_index(np.ones(len(t_preds))*30, t_preds, events)

    if verbose:
        print(f"C-index: {c_index}")       
        #means = [np.mean(t_preds[events == l]) for l in [0, 1]]
        #stds = [np.std(t_preds[events == l]) for l in [0, 1]]
        #plt.bar(['Censored (0)', 'Event (1)'], means, yerr=stds, color=['r', 'g'], alpha=0.7)
        plt.scatter(events, t_preds)
        plt.ylabel("Predicted Time")
        plt.title("Predicted Time by Event Label")
        plt.show()

    return pid_list, events, t_preds

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models", "vitH_gated_surv_2025-05-24", "models", "version_0")
    fyaml = os.path.join(model_dir, "config.yaml")
    if os.path.exists(fyaml):
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    model_path = os.path.join(model_dir, "last.pth")
    model = load_model(config, model_path, config.num_features, long_mode=False)

    # Set these paths to your external data
    base_dir = "/home/fi5666wi/PRIAS_data/features_uni_v2_snd"
    csv_path = "/home/fi5666wi/SND_prostate_cancer/patient_documentation.csv"

    pid_list, events, t_preds, t_true = eval_surv(model.to(device), device, base_dir, csv_path, verbose=True)


