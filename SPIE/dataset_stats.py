from prias_file_reader import PRIAS_Data_Object
import pandas as pd
import os

if __name__ == "__main__":
    Data = PRIAS_Data_Object("/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx", sheet_name=0)
    #some_patients = Data.get_patient_idxs() #[2, 5, 46, 150, 345]

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx"
    sheet = 2
    some_patients = pd.read_excel(xcl_path, sheet_name=sheet)['Patient number']
    N = len(some_patients)
    labels = pd.read_excel(xcl_path, sheet_name=sheet)['act0 treated1']
    ratio = sum(labels) / N

    treated = 0
    active = 0
    base_dir = "/home/fi5666wi/PRIAS_data/wsis"
    for p in some_patients:
        if p in Data.get_patient_idxs():

            wsi_dict = Data.get_patient_data(p)
            label = Data.get_patient_label(p)
            n_visits = 0
            #last = wsi_dict.popitem()
            for n, wsi in enumerate(wsi_dict):
                # If label is 1 (treated), check only last case
                if label == 1 and n+1 < len(wsi_dict):
                    continue
                wsi_paths = [f"{wsi}-{num}_10x.png" for num in wsi_dict[wsi]]
                count = 0
                for wp in wsi_paths:
                    #print(f"{wp} of {p} exists")
                    if int(wp[:2]) < 11:
                        if label == 1:
                            print(f"Warning, Too old slides for {p} ({wp})") # I do not filter these out during training, should I?
                        #break
                    if not os.path.exists(os.path.join(base_dir, wp)):
                        print(f"{wp} does not exist")
                    else:
                        count += 1
                if count > 0:
                    if label == 1:
                        treated += 1
                    else:
                        active += 1
                    n_visits += 1
            print(f"Patient {p}: label: {label}, Visits: {n_visits}")
    
        else:
            print(f"Patient {p} not in file")

    print("--------------------")
    print(f"Total: {N}")
    print(f"Ratio: {ratio}")
    print(f"Active: {active}, Treated: {treated}")