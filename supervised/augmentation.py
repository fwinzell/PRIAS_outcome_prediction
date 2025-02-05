import os
import fnmatch
import random
from scipy import ndimage
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from skimage.io import imread
import math
#import cv2
from torchvision import transforms


def rotate(image, angle):
    [c, w, h] = image.size()
    dim = math.ceil(2 * math.sqrt(h ** 2 / 2))
    exp = dim - h
    # exp has to be divisible by 2
    exp = exp + (exp % 2)

    exp_img = torch.zeros(size=(c, h + exp, h + exp), dtype=image.dtype)
    s = int(exp / 2)
    exp_img[:, s:s + w, s:s + h] = image

    # Mirroring
    left = image[:, 0:s, :]
    left = left.flip(1)  #left[:, ::-1, :]
    exp_img[:, 0:s, s:s + h] = left

    right = image[:, w - s:, :]
    right = right.flip(1)  #right[:, ::-1, :]
    exp_img[:, w + s:, s:s + h] = right

    top = image[:, :, 0:s]
    top = top.flip(2)  #top[:, :, ::-1]
    exp_img[:, s:s + w, 0:s] = top

    bot = image[:, :, h - s:]
    bot = bot.flip(2)  #bot[:, :, ::-1]
    exp_img[:, s:s + w, h + s:] = bot

    # rotated = ndimage.rotate(exp_img, angle, reshape=False)
    # rotated = cv2_rotate_image(exp_img,angle)
    rotated = F.rotate(exp_img, angle, expand=False)
    cropped = rotated[:, s:s + w, s:s + h]

    return cropped


def random_crop_and_resize(img, s=[0.01, 1.0]):
    random_crop = transforms.RandomResizedCrop(size=img.shape[-2:])
    params = random_crop.get_params(torch.tensor(img), scale=s, ratio=[0.95, 1.05])

    #apply = random.choices([True, False], weights=[p, 1-p], k=1)

    cropped_img = F.crop(img, top=params[0], left=params[1], height=params[2], width=params[3])
    img = F.resize(cropped_img, img.shape[-2:])

    return img


def random_rotation_and_flip(img, p=0.5):
    do_flip = random.choices([True, False], weights=[p, 1 - p], k=1)[0]
    if do_flip:
        img = torch.flip(img, dims=[2])  # CHECK DIMENSIONS!!

    rotation_angle = random.choices(np.arange(0, 360, 45), k=1)[0]
    img = rotate(img, int(rotation_angle))

    return img


def hs_distortion(img, sat=(0.6, 1.8), hue=0.05, prob=1.0):
    hs = transforms.ColorJitter(saturation=sat, hue=hue)
    applier = transforms.RandomApply(nn.ModuleList([hs]), p=prob)
    return applier(img)


def color_distortion(img, s=1.0):
    color_jitter = transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
    # random_crop = transforms.RandomResizedCrop(size=image.shape[-2:], scale=(0.05, 1.0))
    #applier = transforms.RandomApply(nn.ModuleList([color_jitter]), p=prob)

    return color_jitter(img)


def gaussian_blur(img, sigma=(0.1, 2.0), prob=0.5):
    blur = transforms.GaussianBlur(kernel_size=25, sigma=sigma)
    applier = transforms.RandomApply(nn.ModuleList([blur]), p=prob)

    return applier(img)

