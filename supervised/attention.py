#import cv2
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PRIAS.heatmaps.wsi_loader import WSIDataset
from PRIAS.supervised.pl.modules import CNNModule
import matplotlib.pyplot as plt

"""def generate_heatmap(dataset, out_array, coords, downsample_factor, patch_sz):
    dim = [int(dataset.shape[0] * downsample_factor), int(dataset.shape[1] * downsample_factor)]
    img = resize(dataset.wsi, dim)
    coords = torch.round(coords * downsample_factor)
    patch_sz = torch.round(patch_sz * downsample_factor)
    heatmap = torch.zeros(img.shape[:2], dtype=torch.float32)
    nummap = torch.zeros(img.shape[:2], dtype=torch.int8)
    for i, prob in enumerate(out_array):
        y, x = coords[i, :]
        x_step = min(img.shape[0] - x, x + patch_sz)
        y_step = min(img.shape[1] - y, y + patch_sz)
        heatmap[x:x + x_step, y:y + y_step] += prob
        nummap[x:x + x_step, y:y + y_step] += 1

    return heatmap/nummap, img"""


def create_heatmap(model, dataloader, shape, patch_sz=299):

    num_map = torch.zeros(shape, dtype=torch.int8)
    score_map = torch.zeros(shape, dtype=torch.float32)
    # Load batches etc
    num_hits = 0
    for i, (data, coords) in enumerate(dataloader):
        with torch.no_grad():
            y_hat = F.sigmoid(model(data))
            prob = y_hat.squeeze()  # want probability [0,1]
            pred = torch.round(y_hat).squeeze()
            num_hits += torch.sum(pred)
            F.threshold(prob, 0.5, 0)
            for j in range(y_hat.size(dim=0)):
                y, x = coords[j, :]
                score_map[x:x + patch_sz, y:y + patch_sz] += prob[j]
                num_map[x:x + patch_sz, y:y + patch_sz] += 1

    attention_map = score_map / torch.max(score_map)
    print("Fraction of malignant patches: {}".format(num_hits / len(dataset)))
    return attention_map


def get_wsi_attention(model, wsi_path, patch_sz=299, overlap=30, batch_size=10):
    dataset = WSIDataset(wsi_path, patch_size=patch_sz, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    atten_scores = torch.zeros(len(dataset))
    pts = torch.zeros((len(dataset), 2))
    for i, (data, coords) in enumerate(dataloader):
        with torch.no_grad():
            y_hat = F.sigmoid(model(data))
            prob = y_hat.squeeze()
            atten_scores[i*batch_size:(i+1)*batch_size] = prob
            pts[i*batch_size:(i+1)*batch_size, :] = coords

    atten_sorted, order = torch.sort(atten_scores, descending=True, stable=True)

    return dataset, atten_sorted, order, pts


if __name__ == "__main__":

    wsi_path = "/home/fi5666wi/Documents/Prostate_images/DOGS training files/PM17 13773-06.png"
    #wsi_path = "/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/images/17PM 30039-02.tiff"
    ckpt_path = os.path.join(os.getcwd(), 'pl/saved_models',
                              'resnet50_2023-03-23', 'lightning_logs', 'version_0', 'checkpoints',
                              'last.ckpt')

    patch_sz = 299
    overlap = 249
    dataset = WSIDataset(wsi_path, patch_size=patch_sz, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=10, drop_last=False)

    #config = parse_config()
    #config.architecture = 'resnet50'
    #model = get_cnn(config)
    #model.load_state_dict(torch.load(ckpt_path), strict=True)
    model = CNNModule.load_from_checkpoint(ckpt_path)
    model.eval()

    """num_map = torch.zeros(dataset.shape[:2], dtype=torch.int8)
    score_map = torch.zeros(dataset.shape[:2], dtype=torch.float32)
    # Load batches etc
    num_hits = 0
    for i, (data, coords) in enumerate(dataloader):
        with torch.no_grad():
            y_hat = F.sigmoid(model(data))
            prob = y_hat.squeeze()# want probability [0,1]
            pred = torch.round(y_hat).squeeze()
            num_hits += torch.sum(pred)
            F.threshold(prob, 0.5, 0)
            for j in range(y_hat.size(dim=0)):
                y, x = coords[j, :]
                score_map[x:x + patch_sz, y:y + patch_sz] += prob[j]
                num_map[x:x + patch_sz, y:y + patch_sz] += 1

    attention_map = score_map/torch.max(score_map)
    print("Fraction of malignant patches: {}".format(num_hits/len(dataset)))"""


    """probs = []
    coords = []
    for i, (data, pts) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            y_hat = torch.sigmoid(model(data))
            #y_hat = torch.round(y_hat, decimals=3)
            probs.append(y_hat.squeeze())  # want probability [0,1]
            coords.append(pts)

    heat_map, wsi = generate_heatmap(dataset, probs, coords, downsample_factor=0.25, patch_sz=patch_sz)"""

    attention_map = create_heatmap(model, dataloader, shape=dataset.shape[:2], patch_sz=patch_sz)
    _, a, order, pts = get_wsi_attention(model, wsi_path)

    w, h = 3, 2
    imfig = plt.figure(99, figsize=(w, h))
    for j in range(6):
        (y,x) = pts[order[j],:].numpy().astype(int)
        patch = dataset.wsi[x:x + patch_sz, y:y + patch_sz, :]
        imfig.add_subplot(h, w, j+1)
        plt.imshow(patch)
        plt.axis('off')


    plt.figure(0)
    plt.imshow(dataset.wsi)
    plt.imshow(attention_map.detach().numpy(), cmap='PuRd', alpha=0.5)
    plt.colorbar()

    plt.show()



