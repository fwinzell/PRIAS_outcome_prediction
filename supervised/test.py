import torch
import os
from torch import nn
import torch.nn.functional as F

from torchvision import models
from torchmetrics import ConfusionMatrix, ROC, AUROC
from prettytable import PrettyTable
from tqdm import tqdm

from image_loaders import get_datasets
#from PRIAS.supervised.pl.modules import CNNModule

from resnet import resnet18, resnet34, resnet50, resnet101, count_parameters
from densenet import densenet201, densenet169
from main import parse_config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

def get_cnn(config):

    n = 1 if config.binary else 4
    if config.architecture == 'resnet18':
        cnn = resnet18(config)
    elif config.architecture == 'resnet34':
        cnn = resnet34(config)
    elif config.architecture == 'resnet50':
        cnn = resnet50(config)
    elif config.architecture == 'resnet101':
        cnn = resnet101(config)
    elif config.architecture == 'densenet169':
        cnn = densenet169(num_classes=n)
    elif config.architecture == 'densenet201':
        cnn = densenet201(num_classes=n)
    else:
        NotImplementedError("Architecture not implemented")

    return cnn


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


def plot_binary(y_pred, y_prob, y_true):
    # Confusion matrix with thresh = 0.5
    confmat = ConfusionMatrix("binary", num_classes=2)
    conf = confmat(y_pred, y_true)
    conf = conf.cpu().numpy()

    print_pretty_table(conf, ["benign", "malignant"])
    cf_fig = get_fancy_confusion_matrix(conf, ["Benign", "Malignant"])

    total_acc = torch.sum(y_pred == y_true) / y_pred.size(dim=0)
    print("Total accuracy: {}".format(total_acc))

    # ROC curve and AUC
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(y_prob, y_true.int())
    auc = AUROC(task="binary")
    auc_val = auc(y_prob, y_true.int())
    print("AUC: {}".format(auc_val))

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


def plot_multiclass(y_pred, y_prob, y_true):
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    conf = confmat(y_pred, y_true)
    conf = conf.cpu().numpy()

    classes = ["benign", "Gleason 3", "Gleason 4", "Gleason 5"]
    print_pretty_table(conf, classes)
    cf_fig = get_fancy_confusion_matrix(conf, classes)

    total_acc = (y_pred == y_true).float().mean()

    print("Total accuracy: {}".format(total_acc))

    # ROC curve and AUC
    roc = ROC(task="multiclass", num_classes=4)
    fpr, tpr, thresholds = roc(y_prob, y_true.int())
    auc = AUROC(task="multiclass", num_classes=4, average="none")
    auc_val = auc(y_prob, y_true.int())
    print("AUC: {}".format(torch.mean(auc_val)))


    plt.figure()
    colors = ['green', 'blue', 'darkblue', 'purple']
    for ii in range(4):
        plt.plot(fpr[ii], tpr[ii], color=colors[ii], lw=1,
                 label=f"ROC {classes[ii]} (auc = {auc_val[ii]})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-1e-2, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    config = parse_config()
    config.binary = False
    config.num_classes = 2 if config.binary else 4
    model_path = os.path.join(config.save_dir,
                              'densenet201_2024-01-26', 'version_1',
                              'best.pth')

    sz = config.patch_size
    #testsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/test_' + str(sz)]

    testsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/test_' + str(sz),
                   '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/test_' + str(sz),
                   '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/test_' + str(sz),
                   '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/test_' + str(sz),
                    '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/test_' + str(sz)]


    _, dataset = get_datasets(testsets, val_split=1.0, binary=config.binary)  # increase val_split to test more images
    model = get_cnn(config)
    model.load_state_dict(torch.load(model_path), strict=True)
    #model = CNNModule.load_from_checkpoint(model_path)
    model.eval()

    batch_size = 10
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0)

    y_pred = torch.zeros(len(dataset))
    y_prob = torch.zeros(len(dataset), 1 if config.binary else 4)
    y_true = torch.zeros(len(dataset))
    test_loop = tqdm(dataloader)
    for i, (images, labels) in enumerate(test_loop):
        with torch.no_grad():
            if config.binary:
                y_hat = torch.sigmoid(model(images))
                y_prob[i * batch_size: (i + 1) * batch_size, ] = y_hat.squeeze()
                y_pred[i * batch_size: (i + 1) * batch_size] = torch.round(y_hat).squeeze()
                y_true[i * batch_size: (i + 1) * batch_size] = labels
            else:
                y_hat = torch.softmax(model(images), dim=1)
                #y_max = torch.max(y_hat, dim=1)
                y_prob[i*batch_size: (i+1)*batch_size, ] = y_hat.squeeze()
                y_pred[i*batch_size: (i+1)*batch_size] = torch.argmax(y_hat, dim=1)
                y_true[i*batch_size: (i+1)*batch_size] = torch.argmax(labels, dim=1)
        test_loop.set_description("Running Test: ")

    if config.binary:
        plot_binary(y_pred, y_prob, y_true)
    else:
        plot_multiclass(y_pred, y_prob, y_true)