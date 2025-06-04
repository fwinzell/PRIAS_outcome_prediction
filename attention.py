import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from models import PRIAS_Model
from dataset import PRIAS_Feature_Dataset, PRIAS_Generic_Dataset
from prias_file_reader import PRIAS_Data_Object

from eval import get_eval_dataset, load_model
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


from train import parse_config
#from supervised.test import get_fancy_confusion_matrix, print_pretty_table
from torchmetrics import ConfusionMatrix, ROC, AUROC
from sklearn.metrics import balanced_accuracy_score
from prettytable import PrettyTable
import seaborn as sn
import yaml
from skimage.io import imread
import h5py

import PIL
PIL.Image.MAX_IMAGE_PIXELS = None


def test_fe(model, device):
    nodes, _ = get_graph_node_names(model)
    # return last layer features and final prediction
    # which level is 'softmax'
    level = nodes.index('softmax')
    fe = create_feature_extractor(model, return_nodes=nodes[level:])
    generic_out = fe(torch.zeros((256,1024)).to(device))
    feature_shape = generic_out[nodes[level]].shape

    print(f"{nodes[level]}: {list(feature_shape)}")

def get_feature_dataset(base_dir,
                     xls_sheet_name=2,
                     use_last_visit=False,
                     use_features=True,
                     use_long_labels=False,
                     include_factors=False,
                     n_month_mode=False,
                     filter_old=True
                     ):
    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    generic = PRIAS_Generic_Dataset(
        path_to_dir=base_dir,
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=xls_sheet_name,
        use_last_visit=use_last_visit,
        use_features=use_features,
        use_long_labels=use_long_labels,
        include_factors=include_factors,
        n_month_mode=n_month_mode,
        filter_old=filter_old
    )

    features = PRIAS_Feature_Dataset(generic.get_slide_dict(), base_dir, train_mode=False, shuffle=False, return_coords=True, return_segmentation=True)

    return features


def get_patches(attention, coord_dict, k=5, wsi_dir="/home/fi5666wi/PRIAS_data/wsis", display=False, patch_size=256):
    top_a, top_i = torch.topk(attention, k)
    bot_a, bot_i = torch.topk(attention, k, largest=False)

    selected_coords = coord_dict['coords'][:, torch.cat((top_i, bot_i)).cpu(), :]

    wsi_idxs = selected_coords[:,:,0].squeeze()
    wsi_idxs, order = torch.sort(wsi_idxs)

    a_vec = torch.cat((top_a, bot_a))[order]
    pts = selected_coords.squeeze()[order, :]

    prev_idx = -1
    patches = []
    for r, idx in enumerate(wsi_idxs):
        if idx != prev_idx:
            wsi_path = os.path.join(wsi_dir, f"{coord_dict['wsi_name'][0]}-{idx}_10x.png")
            wsi = imread(wsi_path)
        patch = wsi[pts[r,2]:pts[r,2]+patch_size, pts[r,1]:pts[r,1]+patch_size, :]
        patches.append(patch)
        prev_idx = idx

        if display:
            plt.imshow(patch)
            plt.title(f"{coord_dict['wsi_name'][0]}-{idx}, Attention: {a_vec[r]}")
            plt.show()

    a_vec, a_order = torch.sort(a_vec, descending=True)
    patches = [patches[i] for i in a_order]
    return patches, a_vec


def save_patches_to_h5(patches, a_vec, wsi_name, pred, prob, label):
    #file_name = f"/home/fi5666wi/Python/PRIAS/patches_attention/{wsi_name}.h5"

    h5_file = h5py.File(f"/home/fi5666wi/Python/PRIAS/patches_attention/{wsi_name}.h5", "w")
    h5_file.create_dataset("patches", data=np.array(patches))
    h5_file.create_dataset("attention", data=a_vec.cpu().numpy())
    h5_file.create_dataset("prediction", data=pred)
    h5_file.create_dataset("probability", data=prob)
    h5_file.create_dataset("label", data=label)
    h5_file.close()


def get_attention(model, device, base_dir, verbose=True, top_k=None, display=False, save_to_h5=True, n_month_mode=False, filter_old=True):
    
    dataset = get_feature_dataset(base_dir, xls_sheet_name=2, use_last_visit=False, use_features=True, n_month_mode=n_month_mode, filter_old=filter_old)

    loader = tqdm(DataLoader(dataset, batch_size=1))

    nodes, _ = get_graph_node_names(model)
    level = nodes.index('softmax')
    fe = create_feature_extractor(model, return_nodes=nodes[level:])


    pids = [int(p) for p in dataset.data_dict['patient_number']]
    slides = [str(s) for s in dataset.data_dict['slides']]
    #attention = torch.zeros(len(slides), n_features)

    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    dataobj = PRIAS_Data_Object(sheet_path, sheet_name=0)


    probs = torch.zeros(len(loader))
    preds = torch.zeros(len(loader))
    true = torch.zeros(len(loader))

    res_dict = {'Gleason Group': [], 'Attention': [], 'Label': [], 'PID': [], 'Slide': [], 'Predicted': []}
    
    with torch.no_grad():
        for i, (pid, feats, label, coord_dict) in enumerate(loader):
            feats, label = feats.to(device), label.to(device)
            output = fe(feats.squeeze())
            probs[i] = output['sigmoid']
            preds[i] = output[nodes[-1]]
            true[i] = label

            if save_to_h5:
                pats, a_vec = get_patches(output['softmax'].squeeze(), coord_dict, k=5, display=display)
                save_patches_to_h5(pats, a_vec, coord_dict['wsi_name'][0], preds[i].item(), probs[i].item(), label.item())
            
            wsi_idxs = coord_dict['coords'][:,:,0].squeeze()
            wsi_idxs, order = torch.sort(wsi_idxs)
            attention = output['softmax'].squeeze()[order]
            if top_k is not None:
                attention, ii = torch.topk(attention, top_k)
                wsi_idxs = wsi_idxs[ii.cpu()]
            n = len(attention)

            gg_dict = dataobj.get_gleason_grade_groups(pid.item(), coord_dict['wsi_name'][0])
            ggs = [gg_dict[int(idx.item())] for idx in wsi_idxs]
            res_dict['Gleason Group'].extend(ggs)
            res_dict['Attention'].extend(attention.cpu().numpy())
            res_dict['Label'].extend([label.item()]*n)

            res_dict['PID'].extend([pid.item()]*n)
            res_dict['Slide'].extend([coord_dict['wsi_name'][0]]*n)
            res_dict['Predicted'].extend([preds[i].item()]*n)

            if verbose:
                print(f"Patient: {pid.item()} \nProbability: {probs[i].item()} Predicition: {preds[i].item()} Label: {label.item()}")
                
    acc = torch.sum(torch.eq(preds, true))/len(loader)

    if verbose:
        print(f"Accuracy: {acc}")

    return pd.DataFrame(res_dict)

def plot_attention(res_df, log=False):
    # Plot attention vs gleason group for each label
    gg = res_df['Gleason Group']
    if log:
        att = np.log(res_df['Attention'])
    else:
        att = res_df['Attention']
    label = res_df['Label']

    plt.figure()
    plt.scatter(gg, att, c=label)
    plt.xlabel("Gleason Group")
    plt.ylabel("Attention")
    plt.title("Attention vs Gleason Group")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_name = "cross_val_2025-05-19" #"vitH_gated_2025-04-28"
    model_dir = os.path.join("/home/fi5666wi/Python/PRIAS/prias_models",
                             model_name, "vitH_gated_0", "models", "version_0")
                             
    #"imagenet201_gated_2024-06-28", "models", "version_0")
    #"vitL_gated_2024-07-01", "models", "version_0")
    #'cross_val_2024-01-15/run_2/models', "version_0",
    #"imagenet201_gated_2024-11-01", "models", "version_0")

    fyaml = os.path.join(model_dir, "config.yaml")
    if os.path.exists(fyaml):
        with open(fyaml, "r") as f:
            config = yaml.safe_load(f)
            config = argparse.Namespace(**config)
    else:
        config = parse_config()

    model_path = os.path.join(model_dir, "best.pth")
    model = load_model(config, model_path, config.num_features, long_mode=False)

    #test_fe(model.to(device), device)
    #config.base_dir = "/home/fi5666wi/PRIAS_data/features_densenet_512"
    #pids, slides, probs, preds, labels = eval(model.to(device), device=device, verbose=False,
    #                            base_dir=config.base_dir)

    res_df = get_attention(model.to(device), device, f"{config.base_dir}", verbose=True, top_k=None, save_to_h5=False, n_month_mode=config.n_month_mode, filter_old=not config.n_month_mode)
    res_df.to_csv(f"/home/fi5666wi/Python/PRIAS/attention_{model_name}_best.csv", index=False)
    plot_attention(res_df, log=True)

    

