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
from dataset import Simple_Feature_Dataset, Simple_Long_Dataset, PRIAS_Feature_Dataset, PRIAS_Generic_Dataset, CrossValidation_Dataset
from prias_file_reader import PRIAS_Data_Object
from cross_validation import parse_config
#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from eval import eval, plot_binary, make_3d_plot, long_eval, get_eval_dataset, load_model, get_fancy_confusion_matrix, print_pretty_table

def validate_cross_val(config, plot=False):
    """
    Evaluate the cross-validation models on validation folds
    Important to use the same seed as during training
    If not, might test on training samples
    """

    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    cv_dataset = CrossValidation_Dataset(
        path_to_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet",
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels_2.xlsx",
        xls_sheet_name=1,
        n_folds=config.K,
        seed=1,
        use_features=config.use_features,
        use_last_visit=config.last_visit_only,
        p_augmentation=config.p_aug,
        use_long_labels=False,
    )

    accs = np.zeros(6)
    aucs = np.zeros(6)
    results = {}
    for i in range(6):
        print(f"##### Version: {i} #####")
        model_path = os.path.join(config.save_dir,
                                  'cross_val_2024-01-15/run_1/models', f"version_{i}",
                                  'densenet201_gated_last.pth')
        model = load_model(config, model_path, 1920, long_mode=False)
        model.to(device)
        _, val_data_fold = cv_dataset.return_splits(K=i)

        loader = tqdm(DataLoader(val_data_fold, batch_size=1))

        probs = torch.zeros(len(loader))
        preds = torch.zeros(len(loader))
        true = torch.zeros(len(loader))

        with torch.no_grad():
            for j, (pid, feats, label) in enumerate(loader):
                feats, label = feats.to(device), label.to(device)
                logits, y_prob, y_hat = model(feats.squeeze())
                probs[j] = y_prob
                preds[j] = y_hat
                true[j] = label

        aucs[i], accs[i] = plot_binary(preds, probs, true, plot)
        results[f"probs_{i}"] = probs
        results[f"true_{i}"] = true

    print(f"Mean Acc: {np.mean(accs)} ({np.std(accs)})")
    print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
    return results


def main_cross_val(config, plot):
    accs = np.zeros(6)
    aucs = np.zeros(6)
    results = {}
    for i in range(6):
        print(f"##### Version: {i} #####")
        model_path = os.path.join(config.save_dir,
                                  'cross_val_2024-06-19/run_0/models', f"version_{i}",
                                  'vitL_gated_last.pth')
        model = load_model(config, model_path, feature_size=1024, long_mode=False) # remember to change feature size
        # make_3d_plot(model, device=torch.device("cpu"), base_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet")
        probs, preds, labels = eval(model.to(device), device=device, verbose=False,
                                    base_dir="/home/fi5666wi/PRIAS_data/features_uni")  # on test data, remember to use the correct
        aucs[i], accs[i] = plot_binary(preds, probs, labels, plot)
        results[f"probs_{i}"] = probs
        results[f"true_{i}"] = labels


    print(f"Mean Acc: {np.mean(accs)} ({np.std(accs)})")
    print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
    return results


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config = parse_config()

    # Eval on cross-validation nets
    #val_res = validate_cross_val(config, plot=True)
    res = main_cross_val(config, plot=True)

    df = pd.DataFrame(res)
    fname = os.path.join("/home/fi5666wi/Documents/PRIAS sheets/ROCresults")
    csv_path = fname + ".csv"
    i = 1
    while os.path.exists(csv_path):
        csv_path = fname + f"_{i}.csv"
        i += 1
    df.to_csv(csv_path, index=False)

