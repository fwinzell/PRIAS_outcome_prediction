import torch
import cv2
import argparse
import os
from torch import nn
import torch.nn.functional as F

from torchvision import models
from torchmetrics import ConfusionMatrix, ROC, AUROC
from prettytable import PrettyTable
from tqdm import tqdm

from wsi_loader import WSIDataset
from torch.utils.data import DataLoader

#from PRIAS.supervised.test import get_cnn
from PRIAS.prias_file_reader import PRIAS_Data_Object
from PRIAS.supervised.densenet import densenet201, densenet169
from post_process import post_process

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
import random
import re


Image.MAX_IMAGE_PIXELS = 3e8 #232 241 570

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default="/home/fi5666wi/PRIAS_data/segmentation_results")
    parser.add_argument("-IN", "--use_imagenet", type=bool, default=False)
    parser.add_argument("--xcl_file", type=str, default="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx")
    parser.add_argument("-x", "--xcl_num", type=int, default=2, help="1 for train, 2 for test, 0 for all")
    parser.add_argument("-r", "--resnet", action='store_true')
    parser.add_argument("--num_features", type=int, default=256)
    parser.add_argument("-c", "--create_stitches", type=bool, default=False)
    parser.add_argument("--csv", type=bool, default=True, help="Write to csv file")
    parser.add_argument("--overlap", type=int, default=75, help="Patch overlap, 75 for 25% overlap, 29 for 10% overlap")
    return parser.parse_args()

def get_tissueSegm(im):
    im = im.astype('float16')#/255
    tissueMorphSizeClose = 39#75#49*1.5#50
    tissueMorphSizeOpen = 39#75#49*1.5#50
    for iCol in range(0,3):
        im[:,:,iCol] = (im[:,:,iCol])/1*2.-1. # [-1,1]
    tissueSegm = np.sum(im,-1)
    tissueSegm[tissueSegm<2] = 0
    tissueSegm[tissueSegm>0] = 1
    tissueSegm = 1-tissueSegm
    segmentation = np.zeros(tissueSegm.shape)
    kernelClose = np.ones((int(tissueMorphSizeClose),int(tissueMorphSizeClose)),np.uint8)
    kernelOpen = np.ones((int(tissueMorphSizeOpen),int(tissueMorphSizeOpen)),np.uint8)
    segmentation[cv2.morphologyEx(cv2.morphologyEx((tissueSegm==1).astype(np.uint8), cv2.MORPH_CLOSE, kernelClose), cv2.MORPH_OPEN, kernelOpen)==1] = 1
    tissueSegm = segmentation
    tissueSegmNew = np.zeros((tissueSegm.shape[0],tissueSegm.shape[1],3))
    for i in range(3):
        tissueSegmNew[:,:,i] = tissueSegm
    return tissueSegmNew

# Score
def score_result(res):
    """
    if sum(res)==0:
        return (0,0)
    a = np.argmax(res)
    if sum(res) == max(res): # Only one class
        b = a
    else:
        b = np.where(res>0)[0]
        if b[-1] == a:
            b = b[-2]
        else:
            b = b[-1]
    """
    noz = np.nonzero(res)[0]
    if len(noz) == 0:
        return (0,0)
    else:
        a = np.argmax(res)
        b = noz[-1]
    
    return (a+3, b+3)

# Percentage results of tissue
def percentage_results(segmTissue,groundTruth):
    totArea = np.sum(segmTissue)
    res = np.zeros((3))
    for i in range(3):
        res[i] = np.sum(groundTruth[:,:,i])*1./totArea * 100
    return res # in %

def area_results(groundTruth):
    res = np.zeros((3))
    for i in range(3):
        res[i] = np.sum(groundTruth[:,:,i])
    return res

def get_cnn():
    n = 4
    cnn = densenet201(num_classes=n)
    return cnn.to(device)

def save_overlay(wsi, seg, downsample_factor, save_name):
    print("Creating overlay image...")

    dim = (int(wsi.shape[1] * downsample_factor), int(wsi.shape[0] * downsample_factor))
    wsi_img = Image.fromarray(wsi)
    wsi_img = wsi_img.resize(dim, resample=Image.BILINEAR)
    wsi_img = wsi_img.convert("RGBA")

    seg_img = Image.fromarray(np.uint8(seg*255))
    seg_img = seg_img.resize(dim, resample=Image.BILINEAR)
    seg_img = seg_img.convert("RGBA")

    overlay = Image.blend(wsi_img, seg_img, alpha=0.5)
    overlay = overlay.convert("RGB")
    overlay.save(save_name)
    print("Done")


def run_precedure_with_maps(wsi_name, path_to_seg):
    #wsi_name = wsi_path.split('/')[-1].split('.')[0]
    segmentation = np.load(path_to_seg)
    area_res = area_results(segmentation)
    grade = score_result(area_res)
    print(f"{wsi_name}: GG {grade[0]}+{grade[1]} with {np.round(area_res,3)}%")

    return area_res, grade

def run_procedure(args, model, wsi_path, save=False):
    torch.cuda.empty_cache()
    dataset = WSIDataset(wsi_path, patch_size=299, overlap=args.overlap)
    loader = DataLoader(dataset, batch_size=5, drop_last=False)

    # num_map keeps track of how many times a pixel has been visited
    num_map = torch.zeros(dataset.shape[:2], dtype=torch.int32)
    # pred_map keeps track of the sum of the predictions for each pixel
    pred_map = torch.zeros(dataset.shape[:2] + (4, ), dtype=torch.float32)
    # Load batches etc
    for i, (imgs, coords) in enumerate(loader):
        output = model(imgs.to(device))  # want softmax score [0,1]
        pred = torch.softmax(output, dim=1)
        for j in range(pred.size(dim=0)):
            y, x = coords[j, :]
            pred_map[x:x + 299, y:y + 299, :] += pred[j].detach().cpu().numpy()
            num_map[x:x + 299, y:y + 299] += 1

    prob_map = torch.div(pred_map, num_map.unsqueeze(-1))
    prob_map = torch.nan_to_num(prob_map, nan=0.0, posinf=0.0, neginf=0.0)
    #filtered_map = post_process(prob_map.numpy(), dataset.get_segmentation())
    tissueSegm = dataset.get_segmentation(value=1)
    segmentation = post_process(prob_map, tissueSegm)

    wsi_name = wsi_path.split('/')[-1].split('.')[0]
    if save:
        # Save a downsampled version of the overlay
        save_overlay(dataset.wsi, segmentation, 0.1,
                   os.path.join(args.save_path, "overlays", f"{wsi_name}_overlay.png"))
        
        # Save the segmentation
        np.save(os.path.join(args.save_path, "maps", f"{wsi_name}_segmentation.npy"), segmentation)
        

    # Results
    perc_res = percentage_results(tissueSegm, segmentation)
    area_res = area_results(segmentation)
    percentage = (sum(perc_res))
    grade = score_result(perc_res)
    print(f"{wsi_name}: GG {grade[0]}+{grade[1]} with {np.round(percentage,3)}%")

    return perc_res, area_res, grade


def check_if_segmentation_exists(args, wsi_path):
    wsi_name = wsi_path.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_path, "maps", f"{wsi_name}_segmentation.npy")
    return os.path.isfile(save_path), wsi_name, save_path


def main(args, wsi_dict, diag_dict, label, model, res_dict={}):
    wsi_dir = "/home/fi5666wi/PRIAS_data/wsis"
    """
    main
    """
    prev_gg, prev_pos = 0, 0
    preds = []
    for j, wsi in enumerate(wsi_dict.keys()):
        y_pred = 0
        cancer_perc = []
        cancer_areas = []
        gleason_grades = []
        if j > 0:
            (prev_gg, prev_pos) = diag_dict[j]

        if label == 1 and j != len(wsi_dict.keys())-1:
            # If label is 1 then only the last visit should be used, otherwise all can be used
            continue
        for index in wsi_dict[wsi]:
            wsi_path = os.path.join(wsi_dir, f"{wsi}-{index}_10x.png")
            seg_done, wsi_name, seg_path = check_if_segmentation_exists(args, wsi_path)
            if seg_done:
                print(f"Segmentation exists for {wsi}-{index}")
                a, gg = run_precedure_with_maps(wsi_name, seg_path)
                cancer_areas.append(a)
                gleason_grades.append(gg)
                res_dict[f"{wsi}-{index}"] = (a, gg)
                continue
            if os.path.isfile(wsi_path):
                p, a, gg = run_procedure(args, model, wsi_path, save=True)
                cancer_perc.append(sum(p))
                cancer_areas.append(a)
                gleason_grades.append(gg)
                res_dict[f"{wsi}-{index}"] = (a, gg)
            else:
                print(f"File {wsi_path} does not exist")

        if len(cancer_perc) == 0:
            continue

        #thr = 5
        # If cancer is less than ~10% consider it noise
        thr = 1200000
        
        y_pred = make_prediction(np.array(cancer_areas), prev_gg, prev_pos, j, thr)

        print(f"Label: {label} Predicted: {y_pred}")

        preds.append(y_pred)
    
    return preds, res_dict


def model_free(args, wsi_dict, diag_dict, label, res_dict={}):

    prev_gg, prev_pos = 0, 0
    preds = []
    for j, wsi in enumerate(wsi_dict.keys()):
        y_pred = 0
        gleason_grades = []
        cancer_areas = []
        if j > 0:
            (prev_gg, prev_pos) = diag_dict[j]

        if label == 1 and j != len(wsi_dict.keys())-1:
            # If label is 1 then only the last visit should be used, otherwise all can be used
            continue
        for index in wsi_dict[wsi]:
            wsi_name = f"{wsi}-{index}"
            path_to_seg = os.path.join(args.save_path, "maps", f"{wsi_name}_10x_segmentation.npy")
            if os.path.isfile(path_to_seg):
                a, gg = run_precedure_with_maps(wsi_name, path_to_seg)
                gleason_grades.append(gg)
                cancer_areas.append(a)
                res_dict[f"{wsi}-{index}"] = (a, gg)
            else:
                print(f"File {wsi_name} does not exist")

        if len(gleason_grades) == 0:
            continue

        thr = np.array([0.0,0.0,0.0])

        y_pred = make_prediction(np.array(cancer_areas), prev_gg, prev_pos, j, thr)

        print(f"Label: {label} Predicted: {y_pred}")

        preds.append(y_pred)
    
    return preds, res_dict
    
def load_and_classify(path, wsi_dict, diag_dict,label):
    res_df = pd.read_csv(path)

    prev_gg, prev_pos = 0, 0
    preds = []
    for j, wsi in enumerate(wsi_dict.keys()):

        if label == 1 and j != len(wsi_dict.keys())-1:
            # If label is 1 then only the last visit is used, so skip to it
            continue

        if j > 0:
            (prev_gg, prev_pos) = diag_dict[j]

        filt = res_df[res_df['WSI'].str.contains(wsi)]
        cancer_area = np.array([tuple(map(float, re.findall(r'\d*\.\d+|\d+', x))) for x in filt['Cancer'].tolist()]) 

        if len(cancer_area) == 0:
            continue

        #gleason_grades = [tuple(map(int, re.findall(r'\d+', x))) for x in filt['Gleason Grade'].tolist()]
        #isup_grades = convert_to_isup(gleason_grades)

        # If cancer is less than 10% consider it noise
        thr = 1200000
        y_pred = make_prediction(np.array(cancer_area), prev_gg, prev_pos, j, thr)

        print(f"Label: {label} Predicted: {y_pred}")
        preds.append(y_pred)

    return preds


def convert_to_isup(gleason_grades):
    """
    Convert a list of Gleason grades (tuples) into ISUP grade groups.

    :param gleason_grades: List of tuples, where each tuple contains two integers representing a Gleason grade.
    :return: A list of integers representing the corresponding ISUP grade groups.
    """
    isup_grades = []
    for grade in gleason_grades:
        primary, secondary = grade
        total_score = primary + secondary

        if total_score <= 6:
            isup_grades.append(1)
        elif total_score == 7:
            if primary == 3:
                isup_grades.append(2)
            else:
                isup_grades.append(3)
        elif total_score == 8:
            isup_grades.append(4)
        else:
            isup_grades.append(5)

    return np.array(isup_grades)


def make_prediction(cancer_areas, prev_gg, prev_pos, j, thr): #thr=np.array([5,5,5])):
    # If cancer is less than theshold consider it noise
    if np.isscalar(thr):
        thr = np.array([thr, thr, thr])
    
    cancer_areas = (cancer_areas > thr) * cancer_areas
    gleason_grades = [score_result(p) for p in cancer_areas]
    isup_grades = convert_to_isup(gleason_grades)

    number_of_positives = sum([1 if x > thr[0] else 0 for x in np.sum(cancer_areas, axis=1)])

    y_pred = 0
    if np.any(isup_grades > 2):
        # If any slide has GG3 or higher, predict positive
        y_pred = 1
    elif prev_gg == 1 and np.any(isup_grades > 1):
        # If a patient goes from GG1 to GG2 or higher, predict positive
        y_pred = 1
    elif j > 0 and number_of_positives > max(2,prev_pos):
        # If more than 2 slides have more than 10% lower grade cancer, and increasing number (i.e not the first visit), predict positive 
        y_pred = 1

    return y_pred



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

# Method to calculate AUC from confusion matrix
def calculate_auc(conf_matrix):
    # Calculate ROC curve
    roc = ROC(num_classes=2)
    roc.update(torch.tensor(conf_matrix))
    auc = roc.compute()
    return auc


if __name__ == '__main__':
    load = True
    use_model = False
    sheet_path = "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx"
    model_path = "/home/fi5666wi/Python/PRIAS/supervised/saved_models/densenet201_2024-01-26/version_1/last.pth"
    data = PRIAS_Data_Object(sheet_path, sheet_name=0)

    args = parse_args()
    xcl_path = args.xcl_file
    list_of_patients = pd.read_excel(xcl_path, sheet_name=args.xcl_num)['Patient number']
    labels = pd.read_excel(xcl_path, sheet_name=args.xcl_num)['act0 treated1']

    #wsi_dict = data.get_patient_data(patient_no=list_of_patients[0])
    #print(wsi_dict)

    #diag_dict = data.get_gg_and_mmcancer(patient_no=list_of_patients[0])
    #print(diag_dict)

    if use_model and not load:
        model = get_cnn()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()

    PRED = []
    TRUE = []
    res_dict = {}
    for i, p in enumerate(list_of_patients):
        print(f"****Patient {p}****")
        wsi_dict = data.get_patient_data(patient_no=p)
        diag_dict = data.get_gleason_diagnosis(patient_no=p)
        if not load:
            if use_model:
                pred, res_dict = main(args, wsi_dict, diag_dict, labels[i], model, res_dict=res_dict)
            else:
                pred, res_dict = model_free(args, wsi_dict, diag_dict, labels[i], res_dict=res_dict)
        else:
            pred = load_and_classify(os.path.join(args.save_path, "results_gleason_grading_51.csv"), wsi_dict, diag_dict, labels[i])
        PRED += pred
        TRUE += [labels[i] for _ in range(len(pred))]
        
    if not load:
        records = [(key, *value) for key, value in res_dict.items()]

        res_df = pd.DataFrame.from_records(records, columns=["WSI", "Cancer", "Gleason Grade"])
        csv_path = os.path.join(args.save_path, "results_gleason_grading.csv")
        while os.path.exists(csv_path):
            csv_path = csv_path.replace(".csv", f"_{random.randint(0,100)}.csv")

        res_df.to_csv(csv_path)


    y_pred, y_true = torch.tensor(PRED), torch.tensor(TRUE)
    res = torch.eq(y_pred, y_true).float()
    print(f"Accuracy: {res.mean().item()}")
    
    confmat = ConfusionMatrix("binary", num_classes=2)
    conf = confmat(y_pred, y_true)
    conf = conf.cpu().numpy()

    print_pretty_table(conf, ["active", "treated"])
    cf_fig = get_fancy_confusion_matrix(conf, ["Active", "Treated"])
    print(f"Sensitivity: {conf[1,1]/(conf[1,1]+conf[1,0])}")
    print(f"Specificity: {conf[0,0]/(conf[0,0]+conf[0,1])}")

    plt.show()












