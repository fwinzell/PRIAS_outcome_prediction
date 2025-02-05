import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

from prias_file_reader import PRIAS_Data_Object
from heatmaps.wsi_loader import WSIDataset
from torch.utils.data import DataLoader
import pandas as pd
import os

wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
data = PRIAS_Data_Object(sheet_path, sheet_name=0)
save_path = "/home/fi5666wi/Documents/PRIAS_data/uni_test" #os.path.join(pat_path, 'patches')

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')


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
        for i, (data, coords) in enumerate(dataloader):
            with torch.no_grad():
                inputs = processor(images=data, return_tensors="pt", do_rescale=False)
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                print(inputs['pixel_values'].shape)
                print(last_hidden_states.shape)
        break
    break

