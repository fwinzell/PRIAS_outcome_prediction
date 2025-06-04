import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from load_slides import SNDSlideDataset
from PRIAS.dataset import PRIAS_Generic_Dataset
from PRIAS.protocol.wsi_loader import WSIDataset
from PRIAS.prias_file_reader import PRIAS_Data_Object
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import PIL
import cv2
from skimage.exposure import match_histograms

PIL.Image.MAX_IMAGE_PIXELS = None

def hsv_transformation(image, mean_hue=156.4, mean_sat=73.7, mean_val=214.0, thresh_val=235):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    thresh = np.moveaxis(np.array([thresh / 255, thresh / 255, thresh / 255]), source=0, destination=-1)
    non_bg_image = (image / 255) * thresh

    hsv = rgb_to_hsv(non_bg_image)

    hhist, _ = np.histogram(hsv[:, :, 0], bins=180)
    shist, _ = np.histogram(hsv[:, :, 1], bins=256)
    vhist, _ = np.histogram(hsv[:, :, 2], bins=256)
    hhist[0], shist[0], vhist[0] = 0, 0, 0
    hmean = np.sum(hhist * range(180)) / np.sum(hhist)
    smean = np.sum(shist * range(256)) / np.sum(shist)
    vmean = np.sum(vhist * range(256)) / np.sum(vhist)

    h_factor = mean_hue/hmean
    s_factor = mean_sat/smean
    v_factor = mean_val/vmean

    hsv[:, :, 0] *= h_factor
    hsv[:, :, 1] *= s_factor
    #hsv[:, :, 2] *= v_factor
    # Also do histogram equaliztion of value channel
    #cdf = histogram_equalization(vhist)
    #plt.figure()
    #plt.plot(cdf)
    #hsv[:, :, 2] = cdf[np.uint8(hsv[:, :, 2]*255)] / 255

    hsv[hsv > 1] = 1
    rgb = hsv_to_rgb(hsv)
    rgb = np.uint8(rgb * 255)
    rgb += np.uint8(-255*thresh+255)

    return rgb

def histmatch(img, ref):
    matched = match_histograms(img, ref, channel_axis=-1)

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True
    )
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(img)
    ax1.set_title('Source')
    ax2.imshow(ref)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()

def hsv_histogram(rgb_image, fig_num=0):
    hsv_image = cv2.cvtColor(np.uint8(rgb_image), cv2.COLOR_RGB2HSV) #cv2.cvtColor(np.uint8(bgr_image), cv2.COLOR_BGR2HSV)

    hue = hsv_image[:, :, 0]
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]

    hhist, _ = np.histogram(hue, bins=180)
    shist, _ = np.histogram(sat, bins=256)
    vhist, _ = np.histogram(val, bins=256)
    hhist[0], shist[0], vhist[0] = 0, 0, 0
    cdf = vhist.cumsum()
    cdf_normalized = cdf * vhist.max() / cdf.max()
    fig = plt.figure(fig_num)
    plt.bar(range(180), hhist, color='magenta', alpha=0.5)
    plt.bar(range(256), shist, color='black', alpha=0.5)
    plt.bar(range(256), vhist, color='lime', alpha=0.5)
    plt.legend(['Hue', 'Saturation', 'Value'])
    plt.plot(cdf_normalized, color='r')

    return fig

def get_snd_patches(batch_size=50):
    name = "patient_077_ABC.mrxs"

    dataset = SNDSlideDataset(name, patch_size=288, overlap=0, segmentation_level=3, magnification_level=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    #dataset.plot_patches_and_segmentation()

    (images, coords) = next(iter(dataloader))
    images = np.moveaxis(images.numpy(), source=1, destination=-1)
    images = np.uint8(images*255)

    """pts = []
    for i, (images, coords) in enumerate(dataloader):
        print('Batch: {}'.format(i))
        pts.append(coords)
        patch = np.moveaxis(images[0].numpy(), source=0, destination=-1)
        patch = np.uint8(patch*255)
        

    dataset.plot_patches_and_segmentation()"""

    return images

def snd_color_distribution(batch_size = 50):
    name = "patient_077_ABC.mrxs"

    dataset = SNDSlideDataset(name, patch_size=288, overlap=0, segmentation_level=3, magnification_level=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    plt.figure(0)
    for (images, coords) in dataloader:
        images = np.moveaxis(images.numpy(), source=1, destination=-1)

        hsv_images = rgb_to_hsv(images)
        hue = hsv_images[:, :, :, 0]
        saturation = hsv_images[:, :, :, 1] 
        value = hsv_images[:, :, :, 2]
        plt.hist(hue.flatten(), bins=256, color='r', alpha=0.5, label='Hue')
        plt.hist(saturation.flatten(), bins=256, color='g', alpha=0.5, label='Saturation')
        plt.hist(value.flatten(), bins=256, color='b', alpha=0.5, label='Value')

    plt.legend()
    plt.title('HSV Histogram')
    #plt.show()


def wsi_color_distribution(batch_size = 50):
    #path = "/home/fi5666wi/PRIAS_data/wsis/13PM 23223-8_10x.png"
    path = "/home/fi5666wi/PRIAS_data/wsis/20PM 20200-2_10x.png"

    dataset = WSIDataset(path, patch_size=288, overlap=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    hue = []
    saturation = []
    value = []
    for (images, coords) in dataloader:
        images = np.moveaxis(images.numpy(), source=1, destination=-1)
        hsv_images = rgb_to_hsv(images)
        hue.append(hsv_images[:, :, :, 0].flatten())
        saturation.append(hsv_images[:, :, :, 1].flatten())
        value.append(hsv_images[:, :, :, 2].flatten())
        
        #plt.hist(hue.flatten(), bins=256, color='r', alpha=0.5, label='Hue')
        #plt.hist(saturation.flatten(), bins=256, color='g', alpha=0.5, label='Saturation')
        #plt.hist(value.flatten(), bins=256, color='b', alpha=0.5, label='Value')

    #plt.show()
    hue = np.concatenate(hue)
    saturation = np.concatenate(saturation)
    value = np.concatenate(value)

    hue_stats = (np.mean(hue), np.std(hue))
    saturation_stats = (np.mean(saturation), np.std(saturation))
    value_stats = (np.mean(value), np.std(value))

    return hue_stats, saturation_stats, value_stats
    

def get_prias_patches(batch_size):

    PDO = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)
    dataset = PRIAS_Generic_Dataset(
        path_to_dir="/home/fi5666wi/PRIAS_data/features_imagenet",
        patient_data_obj=PDO,
        xls_path="/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx",
        xls_sheet_name=3,
        use_last_visit=False,
        use_features=False,
        use_long_labels=False
    )

    tr_dataset = dataset.return_splits(set_split=0)[0]

    trloader = DataLoader(tr_dataset, batch_size=batch_size)

    (pid, images, label) = next(iter(trloader))
    images = np.moveaxis(images.numpy(), source=1, destination=-1)
    images = np.uint8(images*255)  

    return images

def get_wsi_patches(batch_size):
    #path = "/home/fi5666wi/PRIAS_data/wsis/13PM 23223-8_10x.png"
    path = "/home/fi5666wi/PRIAS_data/wsis/20PM 20200-2_10x.png"

    dataset = WSIDataset(path, patch_size=288, overlap=0)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    (images, coords) = next(iter(dataloader))
    images = np.moveaxis(images.numpy(), source=1, destination=-1)
    images = np.uint8(images*255)

    return images


if __name__ == "__main__":

    batch_size = 10
    #snd_color_distribution(batch_size=batch_size)
    h,s,v = wsi_color_distribution(batch_size=batch_size)

    print(f"Mean hue: {h[0]} Sd {h[1]} \nMean saturation: {s[0]} Sd {s[1]} \nMean value: {v[0]} Sd {v[1]}")
    
    snd_imgs = get_snd_patches(batch_size=batch_size)
    prias_imgs = get_wsi_patches(batch_size=batch_size)

    snd_transformed = np.zeros_like(snd_imgs)
    for i in range(batch_size):
        histmatch(snd_imgs[i], prias_imgs[i])
        #fig = hsv_histogram(prias_imgs[i], fig_num=0)
        #fig = hsv_histogram(snd_imgs[i], fig_num=1)
        img_t = hsv_transformation(snd_imgs[i], mean_hue=h[0]*255, mean_sat=s[0]*255, mean_val=v[0]*255, thresh_val=235)
        snd_transformed[i] = img_t
        #fig = hsv_histogram(img_t, fig_num=2)
        #plt.show()

    fig, axs = plt.subplots(3, batch_size, figsize=(20, 10))

    for i in range(batch_size):
        axs[0, i].imshow(snd_imgs[i])
        axs[0, i].axis("off")
        axs[1, i].imshow(prias_imgs[i])
        axs[1, i].axis("off")
        axs[2, i].imshow(snd_transformed[i])
        axs[2, i].axis("off")
    plt.show()



    




