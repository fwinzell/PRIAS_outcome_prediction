import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import yaml
from skimage.io import imread
import h5py


def plot_grid(n=3, correct = True, randomize = True, save = False):
    plt.style.use('ggplot')

    h5_files = os.listdir(main_dir)
    with h5py.File(os.path.join(main_dir, h5_files[0]), 'r') as f:
        n_patches = len(f['patches'])

    if randomize:
        random.shuffle(h5_files)


    fig, axs = plt.subplots(n, n_patches, figsize=(n_patches, n))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    i = 0
    for k,h5_f in enumerate(h5_files):
        if i == n:
            break
        with h5py.File(os.path.join(main_dir, h5_f), 'r') as f:
            patches = f['patches'][:]
            a_vec = np.flip(np.around(np.array(f['attention'][:]),3))
            patches = np.flip(patches, axis=0)

            pred = f['prediction'][()]
            prob = f['probability'][()]
            label = f['label'][()]

            if correct and pred != label:
                continue
            elif not correct and pred == label:
                continue

            # Top should be label = treated        
            if label != 1 and i < n/2:
                continue
            if label == 1 and i >= n/2:
                continue

            for j, patch in enumerate(patches):
                if j == 0:  # Replace the first tile with label and prediction text
                    axs[i, j].text(0.5, 0.5, f"Label: {label}\nPred.: {pred}\nProb.: {prob:.2f}",
                                   horizontalalignment='center', verticalalignment='center', fontsize=10, transform=axs[i, j].transAxes)
                    axs[i, j].axis('off')
                else:
                    axs[i,j].imshow(patch)
                    axs[i,j].set_title(f"{str(a_vec[j])}", fontsize=7)
                    axs[i,j].axis('off')

                    # Add border around the patch
                    if j < n_patches/2: color = 'b' 
                    else: color = 'r'
                    rect = Rectangle((0, 0), patch.shape[1], patch.shape[0], linewidth=2, edgecolor=color, facecolor='none')
                    axs[i, j].add_patch(rect)

            i += 1
    
    plt.tight_layout()
    if save:
        path = "/home/fi5666wi/Python/PRIAS/"
        plt.savefig(f"{path}grid_{n}.png", dpi=500)
    plt.show()



if __name__ == "__main__":
    main_dir = "/home/fi5666wi/Python/PRIAS/patches_attention/"
    plot_grid(n=4, correct = False, randomize = True, save = True)


    