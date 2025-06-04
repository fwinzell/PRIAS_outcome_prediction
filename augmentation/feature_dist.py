import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import norm
import matplotlib.pyplot as plt

from PRIAS.dataset import Simple_Feature_Dataset

def plot_feature_dists():
    label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
                               sheet_name=5)
    dataset = Simple_Feature_Dataset(base_dir="/home/fi5666wi/PRIAS_data/features_imagenet",
                                     data_frame=label_data)
    loader = DataLoader(dataset, batch_size=1)

    mu = torch.zeros(len(loader), 1920)
    sigma = torch.zeros(len(loader), 1920)
    for i, (pid, feature, label) in enumerate(loader):

        mu[i, :] = torch.sum(feature, dim=1)/torch.count_nonzero(feature, dim=1)
        sigma[i, :] = torch.std(feature, dim=1)

    mu_hat = torch.mean(mu, dim=1).numpy()
    sigma_hat = torch.std(sigma, dim=1).numpy()
    max_var = np.max(sigma_hat)
    x_axis = np.linspace(-3 * max_var, 3 * max_var, 500)
    # x_axis = np.arange(-1, 1, 0.0001)

    print(f"Mean-mu: {np.mean(mu_hat)} STD-mu: {np.std(mu_hat)}")
    print(f"Mean-sigma: {np.mean(sigma_hat)} STD-sigma: {np.std(sigma_hat)}")

    N = 100
    cmap = plt.cm.get_cmap('hsv', N)
    for j in range(N):
        pdf = norm.pdf(x_axis, mu_hat[j], sigma_hat[j])
        plt.plot(x_axis, pdf, color=cmap(j))
    plt.show()

def features_vs_norm():
    label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
                               sheet_name=5)
    dataset = Simple_Feature_Dataset(base_dir="/home/fi5666wi/PRIAS_data/features_imagenet",
                                     data_frame=label_data)
    loader = DataLoader(dataset, batch_size=1)

    x_axis = np.arange(-1, 1, 0.0001)
    pdf = norm.pdf(x_axis, 0, 0.01)

    mu = torch.zeros(len(loader), 1920)
    sigma = torch.zeros(len(loader), 1920)
    for i, (pid, feature, label) in enumerate(loader):
        mu[i, :] = torch.mean(feature, dim=1)
        sigma[i, :] = torch.std(feature, dim=1)

    plt.figure()
    plt.plot(x_axis, pdf, color='k')

    N = 100
    cmap = plt.cm.get_cmap('hsv', N)
    for j in range(N):
        plt.plot(mu[j,:], np.zeros(1920), marker='x', color=cmap(j), linestyle="None")
    plt.show()

def feature_histogram():
    label_data = pd.read_excel("/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx",
                               sheet_name=5)
    dataset = Simple_Feature_Dataset(base_dir="/home/fi5666wi/PRIAS_data/features_imagenet",
                                     data_frame=label_data)
    loader = DataLoader(dataset, batch_size=1)

    plt.figure()
    x_axis = np.arange(1,1920)
    for i, (pid, feature, label) in enumerate(loader):
        non_zero_indexes = []
        for j in range(feature.size()[1]):
            #plt.plot(x_axis, feature[0,j,:], marker='x', linestyle="None")
            nonzero = torch.nonzero(feature[0,j,:])
            non_zero_indexes += [x[0] for x in nonzero.numpy()]
            #print(f"{pid} Non-zero: {nonzero.size()[0]}")
            plt.plot(nonzero, feature[0,j,nonzero], marker='x', color='r', linestyle="None")
        non_zero_indexes = np.unique(np.array(non_zero_indexes))
        print(f"{pid} Non-zero: {non_zero_indexes}")

    plt.show()




if __name__ == "__main__":
    plot_feature_dists()
    features_vs_norm()
    feature_histogram()


