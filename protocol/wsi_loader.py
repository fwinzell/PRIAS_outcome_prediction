import cv2
import torch
import os
import numpy as np
import random
import math
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

import matplotlib.pyplot as plt


def tissue_segmentation(wsi, use_otsu=False):
    # This method segments the tissue from the background of a WSI, ignoring holes.
    # It was taken from https://github.com/mahmoodlab/CLAM
    # Input: wsi
    # Output: cv2 Contours object
    def _filter_contours(contours, hierarchy, min_area=255 * 255, max_n_holes=8):
        """
            Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            # print(a)
            # self.displayContours(img.copy(), cont)
            if a == 0: continue
            if min_area < a:
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[:max_n_holes]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > min_area:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    #wsi = cv2.imread(wsi_path)
    img_hsv = cv2.cvtColor(wsi, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], 7)  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, 8, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours

    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    return _filter_contours(contours, hierarchy)  # Necessary for filtering out artifacts


class WSIDataset(Dataset):
    """
    This class is used to load a WSI and extract patches from it.
    It uses the tissue segmentation method from above to ignore background.
    Arguments:
      wsi_path: path to the WSI
      patch_size: size of the patches to extract (a number, the patches will be square)
      overlap: overlap between patches (in pixels)
    """

    def __init__(self,
                 wsi_path,
                 patch_size,
                 overlap):
        self.path = wsi_path
        self.wsi = imread(wsi_path)
        self.shape = self.wsi.shape
        if self.shape[-1] > 3:
            print('Found more than three channels, keeping primary three')
            self.wsi = self.wsi[:,:,:3]
        self.patch_size = patch_size
        self.overlap = overlap
        self.foreground_contours, _ = tissue_segmentation(self.wsi)
        self.patches = self._get_patches()
        self.input_dtype = torch.float32
        self.transform = ToTensor()

    def get_segmentation(self, value=255):
        seg = np.zeros(self.shape[:2], dtype=np.uint8)
        seg = cv2.drawContours(seg, self.foreground_contours, -1, color=(value), thickness=cv2.FILLED)
        return seg

    def _get_patches(self):
        patch_coords = []
        for contour in self.foreground_contours:
            start_x, start_y, w, h = cv2.boundingRect(contour)
            img_h, img_w = self.shape[:2]
            stop_y = min(start_y + h, img_h - self.patch_size + 1)
            stop_x = min(start_x + w, img_w - self.patch_size + 1)

            if stop_x < start_x or stop_y < start_y:
                continue

            step_size_x = math.floor((stop_x - start_x) / math.ceil((stop_x - start_x) / (self.patch_size-self.overlap)))
            step_size_y = math.floor((stop_y - start_y) / math.ceil((stop_y - start_y) / (self.patch_size-self.overlap)))

            x_range = np.arange(start_x, stop_x, step=step_size_x)
            y_range = np.arange(start_y, stop_y, step=step_size_y)
            x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
            coords = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

            for pt in coords:
                cent = pt + int(self.patch_size/2)
                if cv2.pointPolygonTest(contour, tuple(np.array(cent).astype(float)),
                                        measureDist=False) > -1:  # check that point is within contour
                    patch_coords.append(pt)

        return patch_coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        pt = self.patches[index]
        patch = self.wsi[pt[1]:pt[1] + self.patch_size, pt[0]:pt[0] + self.patch_size, :]
        #patch = np.moveaxis(patch.squeeze(), source=-1, destination=0)
        #patch = torch.from_numpy(patch)
        patch = self.transform(patch)
        return patch.type(self.input_dtype), pt

# Run this main method to test the dataset
if __name__ == '__main__':
    path = "/home/fi5666wi/PRIAS_data/wsis/13PM 23223-8_10x.png"

    dataset = WSIDataset(path, patch_size=299, overlap=200)
    dataloader = DataLoader(dataset, batch_size=10, drop_last=False)
    print(len(dataset))
    print(len(dataloader))

    pts = []
    for i, (images, coords) in enumerate(dataloader):
        print('Batch: {}'.format(i))
        pts.append(coords)
        patch = np.moveaxis(images[0].numpy(), source=0, destination=-1)
        patch = np.uint8(patch*255)
        cv2.imshow('Image', patch)
        cv2.waitKey(100)

    wsi = dataset.wsi
    for (x,y) in dataset.patches:
        wsi = cv2.rectangle(wsi, (x,y), (x+256,y+256), color=(0,0,0), thickness=1)
    wsi = cv2.drawContours(wsi, dataset.foreground_contours, -1, color=(0, 255, 0), thickness=cv2.FILLED)

    plt.figure(0)
    plt.imshow(wsi)
    plt.show()


