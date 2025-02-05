import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from extract_patches import PatientObject
from prias_file_reader import PRIAS_Data_Object
import os
import pandas as pd
from heatmaps.wsi_loader import WSIDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from UNI.uni.downstream.extract_patch_features import extract_patch_features_from_dataloader

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
data = PRIAS_Data_Object(sheet_path, sheet_name=0)
save_path = "/home/fi5666wi/Documents/PRIAS_data/uni_test" #os.path.join(pat_path, 'patches')

# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
list_of_patients = pd.read_excel(xcl_path, sheet_name=2)['Patient number']

#for pid in list_of_patients:
#    if pid > 3:
#        break

pid = 9    
wsi_dict = data.get_patient_data(patient_no=pid)
for wsi_id in wsi_dict.keys():
    for num in wsi_dict[wsi_id]:
        print(f"WSI: {wsi_id}-{num}")
        wsi_path = os.path.join(wsi_dir, f"{wsi_id}-{num}_10x.png")

        dataset = WSIDataset(wsi_path, patch_size=288, overlap=75)
        dataloader = DataLoader(dataset, batch_size=10, drop_last=False, shuffle=False)
        #for i, (data, coords) in enumerate(dataloader):
        #    with torch.no_grad():
        #        y_hat = model(data)
        #        print(y_hat.shape)
        features = extract_patch_features_from_dataloader(model, dataloader)


        # convert these to torch
        train_feats = torch.Tensor(features['embeddings'])
        train_labels = torch.Tensor(features['labels']).type(torch.long)
        print(train_feats.shape)
        print(train_labels)
        break
    break





    