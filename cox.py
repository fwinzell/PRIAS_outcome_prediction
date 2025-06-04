import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt

from PRIAS.models import PRIAS_Model
from PRIAS.dataset import PRIAS_Feature_Dataset, PRIAS_Generic_Dataset, CrossValidation_Dataset, PRIAS_Generic_All
from PRIAS.prias_file_reader import PRIAS_Data_Object
#from PRIAS.train import parse_config
from cross_validation import parse_config, seed_torch
from PRIAS.eval import load_model, get_fancy_confusion_matrix, print_pretty_table, plot_binary, plot_time_vs_prob, get_eval_dataset

#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from torchmetrics import ConfusionMatrix, ROC, AUROC
from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable
import seaborn as sn
import yaml


def eval_cross_val(model_dir, config, device, base_dir="/home/fi5666wi/PRIAS_data/features_uni_v2_all"):
    """
    Evaluate the cross-validation models on test data using the ensemble approach.
    Base dir should be without augmentation.
    Set save_to_csv to True to save results to a CSV file (useful for R plotting).
    """
    versions = [os.path.join(model_dir, f"version_{i}", "best.pth") for i in range(6)]
    ensemble = [load_model(config, version, config.num_features, long_mode=False) for version in versions]

    if not hasattr(config, 'gg_mode'):
        config.gg_mode = False

    dataset = get_eval_dataset(base_dir, 
                               xls_sheet_name=2, 
                               use_last_visit=False, 
                               use_features=True, 
                               treated_test=False,
                               n_month_mode=config.n_month_mode,
                               filter_old=True,
                               gg_mode=config.gg_mode,
                               topk=config.k_random_samples if config.gg_mode else 0)

    loader = tqdm(DataLoader(dataset, batch_size=1))

    pids = [int(p) for p in dataset.data_dict['patient_number']]
    slides = [str(s) for s in dataset.data_dict['slides']]
    time = [float(m) for m in dataset.data_dict['time']]

    probs = {pid: [] for pid in pids}
    labels = {}

    for i, model in enumerate(ensemble):
        model.to(device)
        model.eval()
        with torch.no_grad():
            for j, (pid, feats, label) in enumerate(loader):
                feats, label = feats.to(device), label.to(device)
                logits, y_prob, y_hat = model(feats.squeeze())
                probs[pid.item()].append(y_prob.item())
                labels[pid.item()] = label.item()

    mean_probs = torch.tensor([np.mean(probs[pid]) for pid in pids])
    preds = mean_probs > 0.5
    true = torch.tensor([labels[pid] for pid in pids])

    return pids, slides, probs, preds, true, time

    

def eval_all(model, device, base_dir, verbose=True):
    #label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels_2.xlsx", sheet_name=1)
    #label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
    #                           sheet_name=4)
    #dataset = Simple_Feature_Dataset(base_dir=base_dir,
    #                                 data_frame=label_data)
    # sheet names: 0 = all, 1 = train, 2 = test
    dataset = get_cox_eval_dataset(base_dir, xls_sheet_name=2, only_treated=False)

    loader = tqdm(DataLoader(dataset, batch_size=1))

    pids = [int(p) for p in dataset.data_dict['patient_number']]
    slides = [str(s) for s in dataset.data_dict['slides']]

    delta_t = [int(m) for m in dataset.data_dict['time']]

    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))
    if verbose:
        plt.figure()
    with torch.no_grad():
        for i, (pid, feats, label) in enumerate(loader):
            feats, label = feats.to(device), label.to(device)
            logits, y_prob, y_hat = model(feats.squeeze())
            probs[i] = y_prob
            preds[i] = y_hat
            true[i] = label
            if verbose:
                print(f"Patient: {pid.item()} \nProbability: {y_prob.item()} Predicition: {y_hat.item()} Label: {label.item()}")
                #print(torch.count_nonzero(feats))
                plt.scatter(y_prob.item(), label.item(), color='g' if label.item() == y_hat.item() else 'r')
            #pred_sum += torch.eq(y_hat, label).int()
            #like_sum += label.item() * y_prob.item() + (1-label.item()) * (1-y_prob.item())  # Cross-entropy ish
    acc = torch.sum(torch.eq(preds, true))/len(loader)
    #like = like_sum/len(loader)
    if verbose:
        print(f"Accuracy: {acc}")
        #print(f"Likelihood: {like}")
        plt.show()

    return pids, slides, probs, preds, true, delta_t



def get_cox_eval_dataset(base_dir,
                     xls_sheet_name=2,
                     only_treated=False,
                     use_last_visit=False,
                     use_features=True,
                     use_long_labels=False,
                     include_factors=False
                     ):
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    
    dataset = PRIAS_Generic_All(
        path_to_dir=base_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=xls_sheet_name,
        only_treated=only_treated,
        shuffle=False,
        filter_old=False # Keep all visits, even the old
        )
    

    return dataset.return_splits(set_split=0, shuffle=False)[0]


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models", "cross_val_2025-05-29",
                             "vitH_gated_aem_0", "models")
    save_name="cv_vitH_aem"


    fyaml = os.path.join(model_dir, "version_0", "config.yaml")
    if os.path.exists(fyaml): 
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    seed_torch(device, config.seed)
    #results =  validate_cross_val(config, model_dir, plot=False, write_to_excel=True) 
    pids, slides, probs, preds, labels, time = eval_cross_val(config=config, model_dir=model_dir, device=device)

    
    plot_time_vs_prob(probs, time, labels)
    auc, acc, bac = plot_binary(preds, probs, labels)

    print(f"Accuracy: {acc}, BAcc {bac}, AUC: {auc}")
    #a, l = long_eval(model.to(device), device=device, base_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet", verbose=True)

    df = pd.DataFrame({
        "PID": pids,
        "Time": time,
        "Label": labels.cpu().numpy(), 
        "Prob": probs.cpu().numpy()})
    
    df = df.sort_values(by=["PID", "Time"])
    
    print(df)
    fname = f"/home/fi5666wi/R/PRIAS/Cox_{save_name}"
    csv_path = fname + ".csv"
    i = 1
    while os.path.exists(csv_path):
        csv_path = fname + f"_{i}.csv"
        i += 1
    df.to_csv(csv_path, index=False)

