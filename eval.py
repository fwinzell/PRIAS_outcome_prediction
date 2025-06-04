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
from PRIAS.train import parse_config
#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from torchmetrics import ConfusionMatrix, ROC, AUROC
from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable
import seaborn as sn
import yaml


def load_model(config, model_path, feature_size, long_mode):
    #model = PRIAS_Model(config, use_features=config.use_features, feature_size=feature_size, long_mode=long_mode)
    if not hasattr(config, 'time_intervals'):
        config.time_intervals = [0] 
    model = PRIAS_Model(config,
                        use_features=config.use_features,
                        long_mode=config.long_labels,
                        survival_mode=config.survival_mode,
                        hidden_layers=config.hidden_layers,
                        n_follow_up=len(config.time_intervals)+1,
                        feature_size=config.num_features)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    model.eval()
    return model

def get_fancy_confusion_matrix(cf_matrix, classes):
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    return sn.heatmap(df_cm, annot=True).get_figure()


def print_pretty_table(conf, classes):
    conf_table = PrettyTable()
    conf_table.add_column(" ", classes)
    for i, name in enumerate(classes):
        conf_table.add_column(name, conf[:, i])
    print(conf_table)

def plot_binary(y_pred, y_prob, y_true, plot=True):
    # Confusion matrix with thresh = 0.5
    confmat = ConfusionMatrix("binary", num_classes=2)
    conf = confmat(y_pred, y_true)
    conf = conf.cpu().numpy()

    print_pretty_table(conf, ["active", "treated"])
    cf_fig = get_fancy_confusion_matrix(conf, ["Active", "Treated"])

    total_acc = torch.sum(torch.eq(y_pred, y_true)) / y_pred.size(dim=0)
    bal_acc = balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    print("Total accuracy: {}".format(total_acc))
    print("Balanced accuracy: {}".format(bal_acc))

    # ROC curve and AUC
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(y_prob, y_true.int())
    auc = AUROC(task="binary")
    auc_val = auc(y_prob, y_true.int())
    print("AUC: {}".format(auc_val))

    if plot:
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=1,
                 label=f"ROC curve (auc = {auc_val})")

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([-1e-2, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristics')
        plt.legend(loc="lower right")

        plt.show()

    return auc_val, total_acc, bal_acc

def plot_time_vs_prob(y_prob, time, labels):
    plt.figure()
    plt.scatter(time, y_prob, c=labels, cmap='bwr', marker='o')
    plt.colorbar(label='Label')
    plt.axhline(0.5, color='k', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.show()

def make_3d_plot(model, device, base_dir="/home/fi5666wi/PRIAS_data/features_densenet"):

    label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
                               sheet_name=5)
    dataset = Simple_Feature_Dataset(base_dir=base_dir,
                                     data_frame=label_data)
    loader = DataLoader(dataset, batch_size=1)

    Data = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    logits = torch.zeros(len(loader))
    psa_density = torch.zeros(len(loader))
    volumes = torch.zeros(len(loader))
    labels = torch.zeros(len(loader))
    with torch.no_grad():
        for i, (pid, feats, label, _) in enumerate(loader):
            pid = pid.item()
            labels[i] = label
            feats, label = feats.to(device), label.to(device)
            logits[i], y_prob, y_hat = model(feats.squeeze())
            psa, vol = Data.get_psa_and_volume(pid)
            if not np.isnan(psa):
                psa = np.min((psa, 1.0))
                psa_density[i] = psa
            if not np.isnan(vol):
                volumes[i] = vol

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    m = ['o', '^']
    c = ['b', 'r']

    for ii in range(len(loader)):
        label = int(labels[ii])
        ax.scatter(logits[ii], psa_density[ii], volumes[ii], color=c[label], marker=m[label])

    ax.set_xlabel('Logit')
    ax.set_ylabel('PSA density')
    ax.set_zlabel('Prostate volume')

    plt.show()



def eval(model, device, base_dir, verbose=True, treated_test=False, n_month_mode=False, filter_old=True):
    
    # sheet names: 0 = all, 1 = train, 2 = test
    dataset = get_eval_dataset(base_dir, 
                               xls_sheet_name=2, 
                               use_last_visit=False, 
                               use_features=True, 
                               treated_test=treated_test,
                               n_month_mode=n_month_mode,
                               filter_old=filter_old)

    loader = tqdm(DataLoader(dataset, batch_size=1))

    pids = [int(p) for p in dataset.data_dict['patient_number']]
    slides = [str(s) for s in dataset.data_dict['slides']]
    if treated_test:
        delta_t = [int(m) for m in dataset.data_dict['time']]
    elif n_month_mode:
        delta_t = [float(m) for m in dataset.data_dict['delta']]

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
    acc = torch.sum(torch.eq(preds, true))/len(loader)

    if verbose:
        print(f"Accuracy: {acc}")
        plt.show()

    if treated_test or n_month_mode:
        return pids, slides, probs, preds, true, delta_t
    else:
        return pids, slides, probs, preds, true


def get_eval_dataset(base_dir,
                     xls_sheet_name=2,
                     use_last_visit=False,
                     use_features=True,
                     use_long_labels=False,
                     include_factors=False,
                     treated_test=False,
                     n_month_mode=False,
                     filter_old=True,
                     survival_mode=False,
                     gg_mode=False,
                     topk=0
                     ):
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    if treated_test:
        dataset = PRIAS_Generic_All(
            path_to_dir=base_dir,
            patient_data_obj=PDO,
            xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
            xls_sheet_name=xls_sheet_name,
            only_treated=True
        )
    else:
        dataset = PRIAS_Generic_Dataset(
            path_to_dir=base_dir,
            patient_data_obj=PDO,
            xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
            xls_sheet_name=xls_sheet_name,
            use_last_visit=use_last_visit,
            use_features=use_features,
            use_long_labels=use_long_labels,
            include_factors=include_factors,
            filter_old=filter_old, # always filter old slides when evaluating
            n_month_mode=n_month_mode,
            survival_mode=survival_mode,
            gg_dataset=gg_mode,
            top_k_features=topk
        )

    return dataset.return_splits(set_split=0, return_coords=True)[0]


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #model_path = os.path.join(config.save_dir, "imagenet201_gated_2023-07-31", "models", "version_1", "last.pth")
                            # 'cross_val_2023-06-26/run_1/models', 'version_0',
                            # 'densenet201_gated_fold_0last.pth')

    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models", "vitH_gated_2025-05-23", #"vitH_gated_aem_0", 
                             "models", "version_2")
                             
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

    #config.base_dir = "/home/fi5666wi/PRIAS_data/features_densenet_512"
    pids, slides, probs, preds, labels = eval(model.to(device), device=device, verbose=False,
                                base_dir=config.base_dir, treated_test=False)
    #plot_time_vs_prob(probs, delta)
    auc, acc, bac = plot_binary(preds, probs, labels)

    print(f"Accuracy: {acc}, BAcc {bac}, AUC: {auc}")
    #a, l = long_eval(model.to(device), device=device, base_dir="/home/fi5666wi/PRIAS_data/features_lower_densenet", verbose=True)

    df = pd.DataFrame({
        "pids": pids,
        "slides": slides,
        "probs": probs.cpu().numpy(), 
        "preds": preds.cpu().numpy(), 
        "true": labels.cpu().numpy()})
    
    print(df)
    fname = f"/home/fi5666wi/R/PRIAS/Res_{config.architecture}"
    csv_path = fname + ".csv"
    i = 1
    while os.path.exists(csv_path):
        csv_path = fname + f"_{i}.csv"
        i += 1
    df.to_csv(csv_path, index=False)

