import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from prias_file_reader import PRIAS_Data_Object

def count_wsis():
    Data = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx"
    train = pd.read_excel(xcl_path, sheet_name=1)['Patient number']
    test = pd.read_excel(xcl_path, sheet_name=2)['Patient number']

    train_wsis = 0
    test_wsis = 0
    for pid in train:
        wsi_dict = Data.get_patient_data(pid)
        label = Data.get_patient_label(pid)
        num_wsis = [len(wsi_dict[case]) for case in wsi_dict.keys()]
        if label == 1:
            train_wsis += num_wsis[-1]
        elif label == 0:
            train_wsis += sum(num_wsis)

    for pid in test:
        wsi_dict = Data.get_patient_data(pid)
        label = Data.get_patient_label(pid)
        num_wsis = [len(wsi_dict[case]) for case in wsi_dict.keys()]
        if label == 1:
            test_wsis += num_wsis[-1]
        elif label == 0:
            test_wsis += sum(num_wsis)

    print(f"Train: {train_wsis} Test: {test_wsis}, Total: {train_wsis + test_wsis}, Ratio: {train_wsis/(train_wsis + test_wsis)}")

def count_stuff():
    Data = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx"
    train = pd.read_excel(xcl_path, sheet_name=1)['Patient number']
    test = pd.read_excel(xcl_path, sheet_name=2)['Patient number']

    train_wsis = 0
    train_pids = 0
    train_active_visits = 0
    train_treated_visits = 0
    for pid in train:
        wsi_dict = Data.get_patient_data(pid)
        label = Data.get_patient_label(pid)
        t = Data.get_biopsy_timeline(pid, in_months=True)
        
        # Filter out visits older than 2011
        years = np.array(Data.get_biopsy_timeline(pid, in_years=True))
        valid_indices = np.where(years >= 2011)[0]
        t = np.array(t)[valid_indices]

        if len(t) == 0:
            print(f"Patient {pid} has no valid visits after 2011.")
            continue
        train_pids += 1
        
        delta_t = t[-1] - t
        num_wsis = [len(wsi_dict[case]) for case in wsi_dict.keys()]
        num_wsis = np.array(num_wsis)[valid_indices]  # Filter WSIs based on valid indices
        
        if label == 1:
            valid = np.where(delta_t < 30)[0]
            train_wsis += sum(num_wsis[valid])
            train_treated_visits += len(valid)
        elif label == 0:
            num_wsis = num_wsis[:-1]  # Remove last visit
            train_wsis += sum(num_wsis)
            train_active_visits += len(num_wsis)

    test_wsis = 0
    test_pids = 0
    test_active_visits = 0
    test_treated_visits = 0
    for pid in test:
        wsi_dict = Data.get_patient_data(pid)
        label = Data.get_patient_label(pid)
        t = Data.get_biopsy_timeline(pid, in_months=True)

        # Filter out visits older than 2011
        years = np.array(Data.get_biopsy_timeline(pid, in_years=True))
        valid_indices = np.where(years >= 2011)[0]
        t = np.array(t)[valid_indices]

        if len(t) == 0:
            print(f"Patient {pid} has no valid visits after 2011.")
            continue
        test_pids += 1

        delta_t = t[-1] - t
        num_wsis = [len(wsi_dict[case]) for case in wsi_dict.keys()]
        num_wsis = np.array(num_wsis)[valid_indices]  # Filter WSIs based on valid indices

        if label == 1:
            valid = np.where(delta_t < 30)[0]
            test_wsis += sum(num_wsis[valid])
            test_treated_visits += len(valid)
        elif label == 0:
            num_wsis = num_wsis[:-1]  # Remove last visit
            test_wsis += sum(num_wsis)
            test_active_visits += len(num_wsis)

    print(f"Train: PIDs {train_pids} Total {train_active_visits + train_treated_visits} ({train_wsis} WSIs), Active: {train_active_visits}, Treated: {train_treated_visits}")
    print(f"Test: PIDs {test_pids} Total {test_active_visits + test_treated_visits} ({test_wsis} WSIs), Active: {test_active_visits}, Treated: {test_treated_visits}")

def year_distribution():
    Data = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    train = pd.read_excel(xcl_path, sheet_name=3)['Patient number']
    test = pd.read_excel(xcl_path, sheet_name=4)['Patient number']

    y_tr = []
    y_te = []
    for pid in train:
        wsi_dict = Data.get_patient_data(pid)
        last = wsi_dict.popitem()[0]
        y_tr.append(int(last[:2]))

    for pid in test:
        wsi_dict = Data.get_patient_data(pid)
        last = wsi_dict.popitem()[0]
        y_te.append(int(last[:2]))

    mu_tr, sigma_tr = np.mean(y_tr), np.std(y_tr)
    mu_te, sigma_te = np.mean(y_te), np.std(y_te)
    max_var = np.max((sigma_tr, sigma_te))
    # x_axis = np.linspace(-3 * np.min((mu_te, mu_tr)) * max_var, 3 * np.max((mu_te, mu_tr)) * max_var, 500)
    x_axis = np.arange(0, 25, 0.01)

    pdf_tr = norm.pdf(x_axis, mu_tr, sigma_tr)
    pdf_te = norm.pdf(x_axis, mu_tr, sigma_tr)

    plt.figure(1)
    plt.plot(x_axis, pdf_tr, color='b')
    plt.plot(x_axis, pdf_te, color='r')
    plt.plot(y_tr, np.zeros(len(y_tr)), 'xb')
    plt.plot(y_te, np.zeros(len(y_te)), 'xr')

    plt.figure(2)
    plt.hist(y_tr, color='b')
    plt.hist(y_te, color='r')

    print(y_tr)
    print(y_te)

    plt.show()

def risk_factor_analysis():
    Data = PRIAS_Data_Object(
        "/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx",
        sheet_name=0)

    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    pids = pd.read_excel(xcl_path, sheet_name=5)['Patient number']
    status = pd.read_excel(xcl_path, sheet_name=5)['act0 treated1']

    plt.figure()
    colors = ["blue", "red"]
    labels = ["active", "treated"]
    all_values = [[[], []], [[], []]]
    missing_values = np.zeros(len(pids))
    for idx, id in enumerate(pids):
        wsi_dict = Data.get_patient_data(id)
        rfs = Data.get_psa_and_volume(id, all=True)
        label = status[idx]
        print(f"Patient: {id}")
        for i, case in enumerate(wsi_dict.keys()):
            psa,vol = rfs[0][i], rfs[1][i]
            print(f"{case}: psa={psa} volume={vol}")
            if not pd.isna(psa) and not pd.isna(vol):
                plt.scatter(x=vol, y=psa, c=colors[int(label)])
                all_values[label][0].append(psa)
                all_values[label][1].append(vol)
            else:
                missing_values[idx] = 1

    print(f"Missing values: {np.sum(missing_values)}")
    plt.show()

    for j, label in enumerate(labels):
        print(label)
        print(f"PSA (mean,std): ({np.mean(all_values[j][0])},{np.std(all_values[j][0])})")
        print(f"Volume (mean,std): ({np.mean(all_values[j][1])},{np.std(all_values[j][1])})")

if __name__ == "__main__":
    #year_distribution()
    #risk_factor_analysis()
    #count_wsis()
    count_stuff()





