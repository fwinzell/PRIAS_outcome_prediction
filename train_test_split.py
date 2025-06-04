import pandas as pd
import os
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from prias_file_reader import PRIAS_Data_Object
import random


if __name__ == "__main__":

    diff = 1000

    while diff > 0.1:
        seed = random.randint(0,10000) #10
        print(seed)
        xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx"
        patients = pd.read_excel(xcl_path, sheet_name=0)

        pdo = PRIAS_Data_Object("/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx", sheet_name=0)

        active = patients['act0 treated1'] == 0
        active_patients = patients[active]['Patient number']

        treated = patients['act0 treated1'] == 1
        treated_patients = patients[treated]['Patient number']

        train_active = active_patients.sample(frac=0.8, random_state=seed)
        test_active = active_patients.drop(train_active.index)

        train_treated = treated_patients.sample(frac=0.8, random_state=seed)
        test_treated = treated_patients.drop(train_treated.index)

        train = pd.concat([train_active, train_treated])
        test = pd.concat([test_active, test_treated])

        train_df = patients[patients['Patient number'].isin(train)]
        test_df = patients[patients['Patient number'].isin(test)]

        tr_ys = []
        for patient in list(train_df['Patient number']):
            #print(patient)
            bt = pdo.get_biopsy_timeline(patient, years=True)
            tr_ys += bt
        
        te_ys = []
        for patient in list(test_df['Patient number']):
            #print(patient)
            bt = pdo.get_biopsy_timeline(patient, years=True)
            te_ys += bt


        print("mean train years", np.mean(np.array(tr_ys)))
        print("mean test years", np.mean(np.array(te_ys)))
        diff = abs(np.mean(np.array(tr_ys)) - np.mean(np.array(te_ys)))

    print("Done")
    train_df.to_csv(f"/home/fi5666wi/Documents/PRIAS sheets/NEW_train_{seed}.csv", index=False)
    test_df.to_csv(f"/home/fi5666wi/Documents/PRIAS sheets/NEW_test_{seed}.csv", index=False)
