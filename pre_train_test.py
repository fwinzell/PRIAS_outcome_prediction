import torch
import os
import numpy as np
import h5py
from heatmaps.wsi_loader import WSIDataset
from torch.utils.data import DataLoader
# from supervised.attention import get_wsi_attention
import torch.nn.functional as F
from skimage.io import imread
import pandas as pd
import argparse
from argparse import Namespace
import PIL

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from prias_file_reader import PRIAS_Data_Object


PIL.Image.MAX_IMAGE_PIXELS = None

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_features(wsi_path, N, fe, nodes):
    batch_size = 10
    dataset = WSIDataset(wsi_path, patch_size=299, overlap=75)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)


    #atten_scores = torch.zeros(len(dataset), device=device)
    #pts = torch.zeros((len(dataset), 2), device=device)
    features = torch.zeros((len(dataset), N), device=device)
    for i, (data, coords) in enumerate(dataloader):
        data, coords = data.to(device), coords.to(device)
        with torch.no_grad():
            output = fe(data)
            y_hat = torch.sigmoid(output[nodes[-1]])
            prob = y_hat.squeeze()
            print(prob)
            #atten_scores[i * batch_size:(i + 1) * batch_size] = prob
            #pts[i * batch_size:(i + 1) * batch_size, :] = coords
            features[i * batch_size:(i + 1) * batch_size, :] = output[nodes[-2]]

    return features


if __name__ == "__main__":
    wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    data = PRIAS_Data_Object(sheet_path, sheet_name=0)

    save_path = ""

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    xcl_num = 5  # 3 for train, 4 for test, 5 for all
    list_of_patients = pd.read_excel(xcl_path, sheet_name=xcl_num)['Patient number']

    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model = model.to(device)
    nodes, _ = get_graph_node_names(model)
    print(nodes)

    # return last layer features and final prediction
    fe = create_feature_extractor(model, return_nodes=nodes[-2:])

    for pid in list_of_patients:
        print(f"----Patient {pid}----")
        if pid > 5:
            break
        wsi_dict = data.get_patient_data(patient_no=pid)
        all_feats = []

        for wsi_id in wsi_dict.keys():
            print(f'Processing: {wsi_id}')
            for num in wsi_dict[wsi_id]:
                wsi_path = os.path.join(wsi_dir, f"{wsi_id}-{num}_10x.png")
                if os.path.exists(wsi_path):
                    print(f"WSI: {wsi_id}-{num}")
                    f = get_features(wsi_path, 1920, fe, nodes)
                    all_feats += f.tolist()



    print('herå')



