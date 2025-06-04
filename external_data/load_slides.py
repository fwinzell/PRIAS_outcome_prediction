import openslide
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import itertools

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

from PRIAS.protocol.wsi_loader import tissue_segmentation

class SNDSlideDataset(Dataset):
    """
    This class is used to load a WSI and extract patches from it.
    It uses the tissue segmentation method from above to ignore background.
    Arguments:
      wsi_path: path to the WSI
      patch_size: size of the patches to extract (a number, the patches will be square)
      overlap: overlap between patches (in pixels)
    """

    def __init__(self,
                 slide_name,
                 patch_size,
                 overlap,
                 snd_dir="/home/fi5666wi/SND_prostate_cancer/",
                 segmentation_level=2,
                 magnification_level=0):
        self.snd_dir = snd_dir

        self.path = os.path.join(snd_dir, slide_name)
        self.slide = openslide.OpenSlide(self.path)

        self.patch_size = patch_size
        self.overlap = overlap
        
        self.seg_lvl = max(segmentation_level, magnification_level) # cannot be larger than magnification level (lower index = higher resolution)
        self.mag_lvl = magnification_level

        self.magnification = 40/self.slide.level_downsamples[self.mag_lvl]

        self.seg_scale = self.slide.level_downsamples[self.seg_lvl]
        self.mag_scale = self.slide.level_downsamples[self.mag_lvl]
        self.scale_ratio = self.seg_scale/self.mag_scale

        low_res_image = self.slide.read_region((0, 0), 
                                               self.seg_lvl, 
                                               self.slide.level_dimensions[self.seg_lvl])
        low_res_image = low_res_image.convert("RGB")
        low_res_image = np.array(low_res_image)

        self.foreground_contours, _ = tissue_segmentation(low_res_image)
        self.patches = self._get_patch_coords()
        self.patch_inds = list(itertools.accumulate([len(pats) for pats in self.patches.values()]))
        self.input_dtype = torch.float32
        self.transform = ToTensor()
        self.cached_region = -1
        self._load_wsi_region(0)

    def get_segmentation(self, value=255):
        seg = np.zeros(self.slide.level_dimensions[self.seg_lvl][:2], dtype=np.uint8)
        seg = cv2.drawContours(seg, self.foreground_contours, -1, color=(value), thickness=cv2.FILLED)
        return seg
    
    def plot_patches_and_segmentation(self):
        low_res_image = self.slide.read_region((0, 0),
                                               self.seg_lvl,
                                               self.slide.level_dimensions[self.seg_lvl])
        low_res_image = low_res_image.convert("RGB")
        low_res_image = np.array(low_res_image)
        low_res_image = cv2.drawContours(low_res_image, self.foreground_contours, -1, color=(0, 255, 0), thickness=5)

        for key, value in self.patches.items():
            cont = self.foreground_contours[key]
            x, y, _, _ = cv2.boundingRect(cont)
            for pt in value:
                pt = (pt / self.scale_ratio).astype(np.int32) + np.array([x, y])
                size = int(self.patch_size / self.scale_ratio)
                low_res_image = cv2.rectangle(low_res_image, tuple(pt), (pt[0] + size, pt[1] + size), color=(255, 0, 0), thickness=cv2.FILLED)

        plt.figure(0)
        plt.imshow(low_res_image)
        plt.axis("off")
        plt.show()

    def _get_patch_coords(self):
        patch_coords = {}
        for j, contour in enumerate(self.foreground_contours):
            patch_coords[j] = []
            upsampled = (contour * self.scale_ratio).astype(np.int32)

            start_x, start_y, w, h = cv2.boundingRect(upsampled)
            cropped = upsampled - np.array([start_x, start_y])

            stop_y = min(h, h - self.patch_size + 1)
            stop_x = min(w, w - self.patch_size + 1)

            if stop_x < 0 or stop_y < 0:
                continue

            step_size_x = math.floor((stop_x) / math.ceil((stop_x) / (self.patch_size-self.overlap)))
            step_size_y = math.floor((stop_y) / math.ceil((stop_y) / (self.patch_size-self.overlap)))


            x_range = np.arange(0, stop_x, step=step_size_x) #np.arange(start_x, stop_x, step=step_size_x)
            y_range = np.arange(0, stop_y, step=step_size_y) #np.arange(start_y, stop_y, step=step_size_y)
            x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
            coords = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

            for pt in coords:
                cent = pt + int(self.patch_size/2)
                if cv2.pointPolygonTest(cropped, tuple(np.array(cent).astype(float)),
                                        measureDist=False) > -1:  # check that point is within contour
                    patch_coords[j].append(pt)

        return patch_coords
    
    
    def _load_wsi_region(self, index):
        if index == self.cached_region:
            return
        self.cached_region = index
        contour = self.foreground_contours[index]
        upsampled = (contour * self.seg_scale).astype(np.int32)
        x, y, w, h = cv2.boundingRect(upsampled)
        w = max(int(w / self.mag_scale)+1, self.patch_size)
        h = max(int(h / self.mag_scale)+1, self.patch_size)

        # read_region (x,y) in original resolution, (w,h) in magnification resolution??? helt otroligt 
        region = self.slide.read_region((x, y), self.mag_lvl, (w, h))
        
        region = region.convert("RGB")
        self.wsi = np.array(region)

    def display_region(self, index):
        self._load_wsi_region(index)
        plt.imshow(self.wsi)
        plt.axis("off")
        plt.show()


    def __len__(self):
        return self.patch_inds[-1]

    def __getitem__(self, index):
        region_index = np.searchsorted(self.patch_inds, index, side='right')
        self._load_wsi_region(region_index)
        patch_index = index - self.patch_inds[region_index - 1] if region_index > 0 else index
        pt = self.patches[region_index][patch_index]
        patch = self.wsi[pt[1]:pt[1] + self.patch_size, pt[0]:pt[0] + self.patch_size, :]
        
        # Debug
        if patch.shape != (self.patch_size, self.patch_size, 3):
            print(f"Patch shape: {patch.shape}")
            print(f"Index: {index}")
            print(f"Region index: {region_index}")
            print(f"Patch index: {patch_index}")
            print(f"Patch coords: {pt}")
            print(f"Region shape: {self.wsi.shape}")
            self.display_region(region_index)
        
        patch = self.transform(patch)
        return patch.type(self.input_dtype), pt



def lil_test():
    dirpath = "/home/fi5666wi/SND_prostate_cancer/"
    slide_path = os.path.join(dirpath, "patient_001_DEF.mrxs")

    # Open the whole slide image
    slide = openslide.OpenSlide(slide_path)

    # Get image properties
    print("Dimensions:", slide.dimensions)
    print("Levels:", slide.level_count)
    print("Level dimensions:", slide.level_dimensions)

    # Read a region (x, y, width, height) from level 0 (highest resolution)
    #region = slide.read_region((1000, 1000), level=0, size=(512, 512))

    # Convert to a NumPy array (optional)
    #region_np = np.array(region)

    # Show the extracted region
    #region.show()

    level = slide.level_count - 6  
    level_dim = slide.level_dimensions[level]
    print(f"Displaying level {level} with size: {level_dim}")

    # Read the full image at this level
    low_res_image = slide.read_region((0, 0), level, level_dim)
    #high_res_image = slide.read_region((0, 0), 0, slide.dimensions) -> memory error
    low_res_image = low_res_image.convert("RGB")
    low_res_image = np.array(low_res_image)

    #display low res image
    #plt.figure(0)
    #plt.imshow(low_res_image)
    #plt.axis("off")

    fg, _ = tissue_segmentation(low_res_image)

    upsampled_fg = [contour * slide.level_downsamples[level] for contour in fg]

    top_cont = upsampled_fg[0].astype(np.int32)
    x, y, w, h = cv2.boundingRect(top_cont)
    high_res_region = slide.read_region((x, y), 0, (w, h))
    high_res_region = high_res_region.convert("RGB")
    high_res_region = np.array(high_res_region)

    cropped_cont = [top_cont - np.array([x, y])]
    wsi = cv2.drawContours(high_res_region, cropped_cont, -1, color=(0, 255, 0), thickness=10)

    plt.figure(1)
    plt.imshow(wsi)
    plt.axis("off")
    plt.show()

    # Convert to RGB and display
    #low_res_image = low_res_image.convert("RGB")

    #plt.figure(figsize=(10, 10))
    #plt.imshow(low_res_image)
    #plt.axis("off")
    #plt.show()

# Run this main method to test the dataset
def main_test():
    name = "patient_001_DEF.mrxs"

    dataset = SNDSlideDataset(name, patch_size=288, overlap=0, segmentation_level=3, magnification_level=2)
    dataloader = DataLoader(dataset, batch_size=50, drop_last=False)
    print(dataset.magnification)
    print(len(dataset))
    print(len(dataloader))

    pts = []
    for i, (images, coords) in enumerate(dataloader):
        print('Batch: {}'.format(i))
        pts.append(coords)
        patch = np.moveaxis(images[0].numpy(), source=0, destination=-1)
        patch = np.uint8(patch*255)

        #dataset.display_region(i)
        cv2.imshow('Image', patch)
        cv2.waitKey(100)
        if i == 10:
            break

    #wsi = dataset.wsi
    #for (x,y) in dataset.patches:
    #    wsi = cv2.rectangle(wsi, (x,y), (x+256,y+256), color=(0,0,0), thickness=1)
    #wsi = cv2.drawContours(wsi, dataset.foreground_contours, -1, color=(0, 255, 0), thickness=cv2.FILLED)

    #plt.figure(0)
    #plt.imshow(wsi)
    #plt.show()
    dataset.plot_patches_and_segmentation()

if __name__ == "__main__":
    lil_test()


