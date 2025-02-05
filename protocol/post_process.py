import numpy as np
import cv2
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

import matplotlib.pyplot as plt
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

sz = 299

def post_process(res, tissueSegm):
    n_pix = res.shape[0]*res.shape[1]
    if n_pix > 5e7:
        print('Image is too large, halving it...')
        # split image along the largest dimension
        split_dim = np.argmax(res.shape)
        res = np.array_split(res,2,axis=split_dim)
        tissueSegm = np.array_split(tissueSegm,2,axis=split_dim)

        segm = post_process(res[0],tissueSegm[0])
        segm = np.concatenate((segm,post_process(res[1],tissueSegm[1])), axis=split_dim)
    else:
        segm = get_segmentation(res,tissueSegm)
    return segm


def get_segmentation(res,tissueSegm):
    start = time.time()

    ## Smoothen result
    # Convolution with a 50x50 kernel over each channel of the result using torch
    #kernel = torch.ones(4,4,50,50).to(device)/50**2
    #res = torch.tensor(res).to(device)
    #res = F.conv2d(res.permute(2,0,1).unsqueeze(0), kernel, padding="same")
    #res = res.squeeze(0).permute(1,2,0)
    kernel = GaussianBlur(49, sigma=(5, 5))
    res = torch.tensor(res).to(device)
    res = kernel(res.permute(2,0,1).unsqueeze(0))
    res = res.squeeze(0).permute(1,2,0)

    ## Integer result
    segmentation_map = torch.argmax(res, dim=2)
    segmentation_map[tissueSegm == 0] = 0
    segmentation_map = F.one_hot(segmentation_map, num_classes=4)
    segmentation_map = segmentation_map.cpu().numpy().astype(np.uint8)
    segmentation_map = segmentation_map[:, :, 1:]

    ## Remove too small malignant areas
    malignant = np.sum(segmentation_map,-1)
    malignant[malignant>1] = 1
    malignant = malignant.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(malignant, connectivity=4)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    min_size = .4*sz**2 # minimum size of particles we want to keep (number of pixels)
    malignant[:] = 0
    #for every component in the image, you keep it only if it's above min_size
    for iComp in range(0, nb_components):
        if sizes[iComp] >= min_size:
            malignant[output == iComp + 1] = 1
    for i in range(3):
        segmentation_map[malignant==0,i] = 0

    ## Within malignant areas, change grade of too small areas...
    segmentation = np.zeros((res.shape[0],res.shape[1],3))
    for i in range(3): # for each grade
        #find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(segmentation_map[:,:,i], connectivity=4)
        #print('Nbr of components: ', nb_components)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        min_size = .4*sz**2 # minimum size of particles we want to keep (number of pixels)
        #for every component in the image, you keep it only if it's above min_size
        for iComp in range(0, nb_components):
            if sizes[iComp] >= min_size:
                segmentation[output == iComp + 1,i] = 1
            else:
                # To small to keep current grade, pick neighbouring grade
                center = centroids[iComp+1]
                r = .5 #.3
                area = segmentation_map[int(center[1]-sz*r):int(center[1]+sz*r), int(center[0]-sz*r):int(center[0]+sz*r), :]
                tot = np.sum(np.sum(area,0),0)
                newGrade = np.argmax(tot)
                segmentation[output == iComp + 1,newGrade] = 1
    end = time.time()
    print('time: ', end - start)
    return segmentation
