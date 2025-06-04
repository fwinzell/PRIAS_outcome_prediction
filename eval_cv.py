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
import yaml

from models import PRIAS_Model
from dataset import PRIAS_Feature_Dataset, PRIAS_Generic_Dataset, CrossValidation_Dataset
from prias_file_reader import PRIAS_Data_Object
from cross_validation import parse_config, seed_torch
#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from eval import eval, plot_binary, make_3d_plot, get_eval_dataset, load_model, get_fancy_confusion_matrix, print_pretty_table, plot_time_vs_prob

#from collections import Counter

def validate_cross_val(config, model_dir, plot=False, write_to_excel=False):
    """
    Evaluate the cross-validation models on validation folds
    Important to use the same seed as during training
    If not, might test on training samples
    """

    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    
    if not hasattr(config, 'n_month_mode'):
        config.n_month_mode = False
    
    if not hasattr(config, 'gg_mode'):
        config.gg_mode = False
    
    cv_dataset = CrossValidation_Dataset(
        path_to_dir=config.base_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=1,
        n_folds=config.K,
        seed=1,
        use_features=config.use_features,
        use_last_visit=config.last_visit_only,
        p_augmentation=config.p_aug,
        use_long_labels=config.long_labels,
        survival_mode=config.survival_mode,
        n_month_mode=config.n_month_mode,
        long_time_intervals=config.time_intervals,
        filter_old=True,
        gg_dataset=config.gg_mode ,
        top_k_features=config.k_random_samples if config.gg_mode else 0
    )

    accs = np.zeros(config.K)
    aucs = np.zeros(config.K)
    bacs = np.zeros(config.K)
    results = {}
    for i in range(config.K):
        print(f"##### Version: {i} #####")
        model_path = os.path.join(model_dir, f"version_{i}", "best.pth")
        model = load_model(config, model_path, config.num_features, long_mode=(config.long_labels or config.survival_mode))
        model.to(device)
        
        _, val_data_fold = cv_dataset.return_splits(K=i)

        loader = tqdm(DataLoader(val_data_fold, batch_size=1))

        pids = [int(p) for p in val_data_fold.data_dict['patient_number']]
        slides = [str(s) for s in val_data_fold.data_dict['slides']]
        probs = torch.zeros(len(loader))
        preds = torch.zeros(len(loader))
        true = torch.zeros(len(loader))

        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(loader):
                pid, feats, label, = batch
                feats, label = feats.to(device), label.to(device)
                logits, y_prob, y_hat, *a = model(feats.squeeze())
                probs[j] = y_prob
                preds[j] = y_hat
                true[j] = label

        aucs[i], accs[i], bacs[i] = plot_binary(preds, probs, true, plot)
        results[f"pids_{i}"] = pids
        results[f"slides_{i}"] = slides
        results[f"preds_{i}"] = preds
        results[f"probs_{i}"] = probs
        results[f"true_{i}"] = true

        if config.n_month_mode:
            results[f"delta_t_{i}"] = [float(m) for m in val_data_fold.data_dict['delta']]

    if write_to_excel:
        write2excel(model_dir, aucs, accs, bacs)
    else:
        print(f"Mean Acc: {np.mean(accs)} ({np.std(accs)})")
        print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
        print(f"Mean BAC: {np.mean(bacs)} ({np.std(bacs)})")
    return results

def write2excel(model_dir, aucs, accs, bacs):
    table = PrettyTable()
    table.field_names = ["AUC", "AUC std", "BAC", "BAC std", "Acc", "Acc std"]

    table.add_row([np.round(np.mean(aucs), 3), np.round(np.std(aucs), 3),
                   np.round(np.mean(bacs), 3), np.round(np.std(bacs), 3),
                   np.round(np.mean(accs), 3), np.round(np.std(accs), 3)])
    print(table)

    df = pd.DataFrame(table.rows, columns=table.field_names)
    fname = f"{os.path.dirname(model_dir)}/CV_val.xlsx"

    # Check if the file already exists
    if os.path.exists(fname):
        # Read the existing file
        existing_df = pd.read_excel(fname)
        # Append the new data
        df = pd.concat([existing_df, df], ignore_index=True)

    # Save the updated DataFrame back to the file
    df.to_excel(fname, index=False)
    print(f"Results appended to {fname}")



def main_cross_val(model_dir, config, device, base_dir="/home/fi5666wi/PRIAS_data/features_uni_v2_all", save_to_csv=False, save_name="model"):
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

    results = {}

    for i, model in enumerate(ensemble):
        model.to(device)
        model.eval()
        pids = []
        probs = []
        times = []
        deltas = []
        slides = []
        labels = []
        with torch.no_grad():
            for j, (pid, feats, label, ret_dict) in enumerate(loader):
                feats, label = feats.to(device), label.to(device)
                _, y_prob, _ = model(feats.squeeze())
                pids.append(pid.item())
                probs.append(y_prob.item())
                slides.append(ret_dict['wsi_name'][0])
                times.append(ret_dict['time'][0].item())
                deltas.append(ret_dict['delta'][0].item())
                labels.append(label.item())
        results[f"model_{i}"] = {
            "pids": pids,
            "probs": probs,
            "slides": slides,
            "times": times,
            "deltas": deltas,
            "labels": labels
        }
        
    pids = np.array(results["model_0"]["pids"])
    slides = np.array(results["model_0"]["slides"])
    times = np.array(results["model_0"]["times"])
    deltas = np.array(results["model_0"]["deltas"])
    labels = torch.tensor(results["model_0"]["labels"])

    all_probs = np.array([results[f"model_{i}"]["probs"] for i in range(len(ensemble))])
    mean_probs = np.mean(all_probs, axis=0)
    mean_probs = torch.tensor(mean_probs)
    preds = mean_probs > 0.5
    
    auc, acc, bac = plot_binary(preds, mean_probs, labels, plot=True)
    print(f"Accuracy: {acc}, BAcc {bac}, AUC: {auc}")

    if save_to_csv:
        df = pd.DataFrame({
            "pids": pids,
            "slides": slides,
            "probs": mean_probs.cpu().numpy(), 
            "preds": preds.cpu().numpy(), 
            "true": labels.cpu().numpy()}
            )
        
        if config.n_month_mode:
            df["delta_t"] = deltas
            df["time"] = times
    
        fname = f"/home/fi5666wi/R/PRIAS/Res_{save_name}"
        csv_path = fname + ".csv"
        i = 1
        while os.path.exists(csv_path):
            csv_path = fname + f"_{i}.csv"
            i += 1
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def main_cross_val_old(config, model_dir, plot):
    """
    Old function, do not use
    """
    accs = np.zeros(config.K)
    aucs = np.zeros(config.K)
    bacs = np.zeros(config.K)
    results = {}
    
    for i in range(config.K):
        print(f"##### Version: {i} #####")
        model_path = os.path.join(model_dir, f"version_{i}", "best.pth")
        model = load_model(config, model_path, config.num_features, long_mode=(config.long_labels or config.survival_mode))
        
        pids, slides, probs, preds, labels, *delta_t = eval(model.to(device), device=device, verbose=False,
                                base_dir="/home/fi5666wi/PRIAS_data/features_uni_v2_all", treated_test=False, n_month_mode=config.n_month_mode, filter_old= not config.n_month_mode)

        if plot:
            aucs[i], accs[i], bacs[i] = plot_binary(preds, probs, labels)

        results[f"pids_{i}"] = pids
        results[f"slides_{i}"] = slides
        results[f"preds_{i}"] = preds
        results[f"probs_{i}"] = probs
        results[f"true_{i}"] = labels

        if config.n_month_mode:
            results[f"delta_t_{i}"] = delta_t

    if plot:
        print(f"Mean Acc: {np.mean(accs)} ({np.std(accs)})")
        print(f"Mean AUC: {np.mean(aucs)} ({np.std(aucs)})")
        print(f"Mean BAC: {np.mean(bacs)} ({np.std(bacs)})")
    return results


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models", "cross_val_2025-05-27",
                             "vitH_gated_aem_2", "models")


    fyaml = os.path.join(model_dir, "version_0", "config.yaml")
    if os.path.exists(fyaml): 
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    seed_torch(device, config.seed)
    #results =  validate_cross_val(config, model_dir, plot=False, write_to_excel=True) 
    results = main_cross_val(config=config, model_dir=model_dir, device=device, save_to_csv=True, save_name="cv_vitH_aem")
    
    """
    max_length = max([len(results[f"pids_{ii}"]) for ii in range(config.K)])
    padded_res = {key: list(values) + [""] * (max_length - len(values)) for key, values in results.items()}
    df = pd.DataFrame(padded_res)
    
    #print(df)
    fname = f"/home/fi5666wi/R/PRIAS/CV_res_{config.architecture}"
    csv_path = fname + ".csv"
    i = 1
    while os.path.exists(csv_path):
        csv_path = fname + f"_{i}.csv"
        i += 1
    #df.to_csv(csv_path, index=False)
    """


    try:
        if config.n_month_mode:
            for k in range(config.K):
                plot_time_vs_prob(results[f"probs_{k}"], results[f"delta_t_{k}"], results[f"true_{k}"])
    except Exception as e:
        print(f"Error plotting time vs prob: {e}")

