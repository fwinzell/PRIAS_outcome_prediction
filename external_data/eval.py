import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt

from PRIAS.models import PRIAS_Model
from PRIAS.eval import load_model, plot_binary, print_pretty_table, get_fancy_confusion_matrix
from PRIAS.train import parse_config

#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from torchmetrics import ConfusionMatrix, ROC, AUROC
from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable
import seaborn as sn
import yaml
import h5py


class SND_Feature_Dataset(Dataset):
    def __init__(self,
                 path_to_dir,
                 csv_path,
                 pids=None,
                 filter_isup=0,
                 gg_mode=False,
                 top_k=256):
        super(SND_Feature_Dataset, self).__init__()
        self.path = path_to_dir
        self.patient_df = pd.read_csv(csv_path)
        self.gg_mode = gg_mode
        self.top_k = top_k # Only used if gg_mode is True
        if pids is not None:
            self.patient_df = self.patient_df[self.patient_df['patient_n'].isin(pids)]
        if filter_isup > 0:
            self.patient_df = self.patient_df[(self.patient_df['isup'] >= filter_isup) | (self.patient_df['isup'] == 0)]

    def get_pids(self):
        return self.patient_df['patient_n'].tolist()

    def __len__(self):
        return len(self.patient_df)
    
    def __getitem__(self, idx):
        row = self.patient_df.iloc[idx]
        pid = row['patient_n']
        label = row['positive']
        isup = row['isup']
        psa = row['psa']

        with h5py.File(os.path.join(self.path, f"{pid}.h5"), 'r') as h5df_file:
            if torch.is_tensor(h5df_file["features"][0]):
                feats = h5df_file["features"][:]
            else:
                feats = torch.Tensor(h5df_file["features"][:])
            
            if self.gg_mode:
                gg_probs = h5df_file["gg_score"][:]
                order = np.argsort(np.squeeze(-gg_probs)) # Sort by GG score, descending
                feats = feats.squeeze()[order, :]
                feats = feats[:self.top_k, :]

        return pid, feats, label, isup, psa
    

def eval(model, device, base_dir, verbose=True):
    pids = [os.path.splitext(f)[0] for f in os.listdir(base_dir)]

    dataset = SND_Feature_Dataset(base_dir, "/home/fi5666wi/SND_prostate_cancer/patient_documentation.csv", pids)

    loader = tqdm(DataLoader(dataset, batch_size=1))

    #pids = [int(p) for p in dataset.data_dict['patient_number']]
    #slides = [str(s) for s in dataset.data_dict['slides']]
    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))
    if verbose:
        plt.figure()
    with torch.no_grad():
        for i, (pid, feats, label, isup, psa) in enumerate(loader):
            feats, label = feats.to(device), label.to(device)
            logits, y_prob, y_hat = model(feats.squeeze())
            probs[i] = y_prob
            preds[i] = y_hat
            true[i] = label
            if verbose:
                #print(f"Patient: {pid[0]} \nProbability: {y_prob.item()} Predicition: {y_hat.item()} Label: {label.item()}")
                plt.scatter(y_prob.item(), label.item(), color='g' if label.item() == y_hat.item() else 'r')
    acc = torch.sum(torch.eq(preds, true))/len(loader)
    if verbose:
        print(f"Accuracy: {acc}")
        plt.show()


    return dataset.patient_df['patient_n'], probs, preds, true


def eval_cross_validation(model_dir, config, base_dir, device, gg_mode=False, top_k=256, save_to_csv=False, save_name="bob"):
    versions = [os.path.join(model_dir, f"version_{i}", "best.pth") for i in range(6)]
    ensemble = [load_model(config, version, config.num_features, long_mode=False) for version in versions]

    pids = [os.path.splitext(f)[0] for f in os.listdir(base_dir)]
    dataset = SND_Feature_Dataset(base_dir, "/home/fi5666wi/SND_prostate_cancer/patient_documentation.csv", pids, filter_isup=1, 
                                  gg_mode=gg_mode, top_k=top_k)
    pids = dataset.get_pids()
    loader = tqdm(DataLoader(dataset, batch_size=1))

    probs = {pid: [] for pid in pids}
    labels = {}

    for i, model in enumerate(ensemble):
        model.to(device)
        model.eval()
        with torch.no_grad():
            for j, (pid, feats, label, isup, psa) in enumerate(loader):
                feats, label = feats.to(device), label.to(device)
                logits, y_prob, y_hat = model(feats.squeeze())
                probs[pid[0]].append(y_prob.item())
                labels[pid[0]] = label.item()

    mean_probs = torch.tensor([np.mean(probs[pid]) for pid in pids])
    preds = mean_probs > 0.5
    true = torch.tensor([labels[pid] for pid in pids])

    auc, acc, bac = plot_binary(preds, mean_probs, true, plot=True)
    print(f"Accuracy: {acc}, BAcc {bac}, AUC: {auc}")

    if save_to_csv:
        df = pd.DataFrame({
            "pids": pids,
            "probs": mean_probs.cpu().numpy(), 
            "preds": preds.cpu().numpy(), 
            "true": true.cpu().numpy()}
            )
    
        fname = f"/home/fi5666wi/R/PRIAS/Res_{save_name}"
        csv_path = fname + ".csv"
        i = 1
        while os.path.exists(csv_path):
            csv_path = fname + f"_{i}.csv"
            i += 1
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def eval_survival(model, config, device, base_dir):
    return None


def main(model_dir, config, device):
    model_path = os.path.join(model_dir, "best.pth")
    model = load_model(config, model_path, config.num_features, long_mode=False)

    pids, probs, preds, labels = eval(model.to(device), device=device, verbose=True,
                                base_dir="/home/fi5666wi/PRIAS_data/features_uni_v2_snd")
    auc, acc, bac = plot_binary(preds, probs, labels)

    print(f"Accuracy: {acc}, BAcc {bac}, AUC: {auc}")
    #a, l = long_eval(model.to(device), device=device, base_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet", verbose=True)

    df = pd.DataFrame({
        "pids": pids,
        "probs": probs.cpu().numpy(), 
        "preds": preds.cpu().numpy(), 
        "true": labels.cpu().numpy()})
    
    print(df)

    fname = f"/home/fi5666wi/R/PRIAS/Res_SND_ext"
    csv_path = fname + ".csv"
    i = 1
    while os.path.exists(csv_path):
        csv_path = fname + f"_{i}.csv"
        i += 1
    #df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models", "cross_val_2025-05-29",
                             "vitH_gated_gg_0", "models")
                             
    #"vitL_gated_2024-12-09", "models", "version_0")
    #"imagenet_gated_2024-12-09", "models", "version_0")
    #"densenet_gated_2024-12-09", "models", "version_0")

    fyaml = os.path.join(model_dir, "version_0", "config.yaml")
    if os.path.exists(fyaml):
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    eval_cross_validation(model_dir, config, "/home/fi5666wi/PRIAS_data/features_uni_v2_snd_gg", device, gg_mode=config.gg_mode, top_k=256,
                          save_to_csv=False, save_name="ext_vitH_gg")

    
    




