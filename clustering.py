import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from prias_file_reader import PRIAS_Data_Object


if __name__ == "__main__":
    Data = PRIAS_Data_Object("/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx", sheet_name=0)
    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    label_data = pd.read_excel(xcl_path, sheet_name=5)

    psa = []
    vol = []
    ggs = []
    mmc = []
    labels = []
    for i,pid in enumerate(label_data["Patient number"]):
        p, v = Data.get_psa_and_volume(pid, all=False, last=True)
        g, m = Data.get_gg_and_mmcancer(pid, all=False, last=True)
        label = label_data["act0 treated1"][i]
        if label == 1:
            print(f"Patient {pid} GG: {g} Label: {label}")
        nan_idx = []
        for j in range(len(p)):
            if np.isnan(p[j]) or np.isnan(v[j]) or np.isnan(g[j]) or np.isnan(m[j]):
                nan_idx.append(j)
        p = np.delete(p, nan_idx)
        v = np.delete(v, nan_idx)
        g = np.delete(g, nan_idx)
        m = np.delete(m, nan_idx)

        psa += list(p)
        vol += list(v)
        ggs += list(g)
        mmc += list(m)
        labels += list(np.ones(len(p))*int(label_data["act0 treated1"][i]))

    labels = np.array(labels)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    m = ['o', '^']
    c = ['g', 'r']

    for ii in range(len(psa)):
        label = int(labels[ii])
        ax.scatter(psa[ii], vol[ii], ggs[ii], color=c[label], marker=m[label])

    ax.set_zlabel('Gleason grade group')
    ax.set_xlabel('PSA density')
    ax.set_ylabel('Prostate volume')

    plt.show()

    #K means
    X = np.array([psa, vol, ggs, mmc]).T
    kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(X)
    total_acc = np.zeros(2)
    for k in range(2):
        l = labels[kmeans==k]
        _, counts = np.unique(l, return_counts=True)
        total_acc[0] += counts[k]
        total_acc[1] += counts[1-k]
        print(f"Cluster {k}: counts:{counts}")
        #total_acc += acc

    print(f"Total accuracy: {total_acc/len(kmeans)}")


