import pandas as pd
import os
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class PRIAS_Data_Object(object):
    """
    This is an object that contains all the data from the PRIAS excel sheet
    It seperates useful information into different dictionaries
    Input:
        file_path: path to the excel sheet
        sheet_name: which sheet to read from (should be 0)
    Output:
        info_dict: keys: patient numbers,
                   values: [biopsy ids of this patient, dates for biopsies, label (active/treatment)]
        slide_dict: keys: patient numbers, values: [[list of all slides], [datetime of scanned slides]]
        diag_dict: keys: patient numbers,
                   values: gleason grade group, mmcancer, psa, pros_volume, label (active/treatment)
    """

    def __init__(self, file_path, sheet_name=0):
        self.file_path = file_path

        self.prias_data = pd.read_excel(file_path, sheet_name=sheet_name)
        self.slide_dict = {}
        self.info_dict = {}
        self.diag_dict = {}

        self.patients = []
        info = [[], []]  # biopsies, dates, (label)
        diagnosis = [[], [], [], []]  # (gleason grade group, number of positive), mmcancer, psa, pros_volume, (label)
        slides = [[], [], []]  # slide names, scanned date, gleason grade group
        biopsy = None
        mmc = 0
        gg = 0
        n_pos = 0
        for i, p in enumerate(self.prias_data['prias patient no']):
            if ispat(p):  # is p corresponding with a new patient?
                if slides[0]:  # if slides is not empty, store in dict for previous patient before overwriting
                    self.slide_dict[str(current_pat_no)] = slides
                    diagnosis[0].append((gg, n_pos))
                    diagnosis[1].append(mmc)
                    self.patients.append(current_pat_no)
                if info[0]:
                    self.info_dict[str(current_pat_no)] = info
                if diagnosis[0]:
                    self.diag_dict[str(current_pat_no)] = diagnosis

                #print('Patient number {}'.format(p))
                current_pat_no = p

                # Get all relevant info from the first row
                # get biopsy id
                biopsy = str(self.prias_data['biopsies'][i]).split(":")[0]
                # get biopsy date if it exists
                date = self.prias_data['date of biopsy MMDDYY'][i]
                if type(date) != datetime: date = None
                # PSA and prostate volume
                psa = self.check_num(self.prias_data['PSA'][i])
                vol = self.check_num(self.prias_data['Prost Vol'][i])
                # label (active/treatment)
                label = int(self.prias_data['0act 1treated 3other '][i])
                gg = 0  # self.gleason_score(self.prias_data['diagnosis'][i])
                n_pos = 0
                slides = [[], [], []]
                info = [[], []]
                diagnosis = [[], [], [], []]
                mmc = 0
                n_slides = 0
                info.append(label)
                diagnosis.append(label)
                # slides.append(biopsy + "_1")
                # diag.append(prias_data['diagnosis'][i])
            else:  # not a new patient
                if self.prias_data['number'][i] == 1:  # update biopsy if needed
                    biopsy = str(self.prias_data['biopsies'][i]).split(":")[0]
                    psa = self.check_num(self.prias_data['PSA'][i])
                    vol = self.check_num(self.prias_data['Prost Vol'][i])
                    date = self.prias_data['date of biopsy MMDDYY'][i]
                    if type(date) != datetime: date = None
                    info[0].append(biopsy)
                    info[1].append(date)
                    if n_slides != 0:
                        diagnosis[0].append((gg,n_pos)) # /nslides
                        diagnosis[1].append(mmc)
                    diagnosis[2].append(psa)
                    diagnosis[3].append(vol)
                    mmc = 0
                    gg = 0
                    n_slides = 0
                    n_pos = 0
            if pd.notna(self.prias_data['number'][i]):
                if not isinstance(self.prias_data['number'][i], str):
                    if not info[0]: info[0].append(biopsy), info[1].append(date)  #, info[3].append(psa), info[4].append(vol),
                    if not diagnosis[2]: diagnosis[2].append(psa), diagnosis[3].append(vol)
                    # Slide id
                    slides[0].append(biopsy + "_" + str(self.prias_data['number'][i]))
                    n_slides += 1

                    # Scanned date
                    scan_date = self.prias_data['scanned'][i]
                    if type(scan_date) == str:
                        if scan_date[0].isdigit():
                            scan_date = datetime.strptime(scan_date, "%d/%m/%Y")
                        else:
                            scan_date = None
                    elif type(scan_date) == datetime:
                        str_date = datetime.strftime(scan_date, "%m/%d/%Y")
                        scan_date = datetime.strptime(str_date, "%d/%m/%Y")
                    slides[1].append(scan_date)

            if pd.notna(self.prias_data['mm cancer'][i]):
                mmc += getnum(self.prias_data['mm cancer'][i])

            this_gg = self.get_gleason_gg(self.prias_data['diagnosis'][i])
            slides[2].append(this_gg)
            gg = max(gg, this_gg)
            if this_gg > 0:
                n_pos += 1

        if slides[0]:  # Save last patient also
            self.slide_dict[str(current_pat_no)] = slides
            diagnosis[0].append((gg,n_pos))
            diagnosis[1].append(mmc)
            self.patients.append(current_pat_no)
        if info[0]:
            self.info_dict[str(current_pat_no)] = info
        if diagnosis[0]:
            self.diag_dict[str(current_pat_no)] = diagnosis

    def len(self):
        return len(self.patients)

    def get_patient_idxs(self):
        return self.patients

    def get_patient_data(self, patient_no):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        wsi_dict = {}
        keys = self.info_dict[str(patient_no)][0]
        slides = np.array(self.slide_dict[str(patient_no)][0])
        for n in keys:
            k = get_slide_name(n)
            vals = slides[np.array([n in s for s in self.slide_dict[str(patient_no)][0]])]
            trimmed = [vals[i].split('_')[1] for i in range(len(vals))]
            while 'nan' in trimmed: trimmed.remove('nan')
            wsi_dict[k] = [int(v) for v in trimmed]

        return wsi_dict

    def get_patient_label(self, patient_no):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        return self.info_dict[str(patient_no)][-1]

    def get_scanned_dates(self, patient_no, format="%Y-%m-%d"):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        scan_dict = {}
        dates = []
        for x in self.slide_dict[str(patient_no)][1]:
            if type(x) == datetime:
                    dates.append(x.strftime(format))
            else:
                dates.append(np.nan)  
        keys = self.info_dict[str(patient_no)][0]
        for n in keys:
            k = get_slide_name(n)
            scan_dict[k] = np.array(dates)[np.array([n in s for s in self.slide_dict[str(patient_no)][0]])]
            scan_dict[k] = np.unique(scan_dict[k], equal_nan=True)

        return scan_dict

    def get_biopsy_timeline(self, patient_no, in_years=False, in_months=False):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        dates = []
        years = []
        for j,x in enumerate(self.info_dict[str(patient_no)][1]):
            if x is None:
                yy = int('20' + self.info_dict[str(patient_no)][0][j][:2])
                x = datetime(year=yy, month=1, day=1)
                if j > 0:
                    x = max(x, dates[-1])
            dates.append(x)
            years.append(x.year)
        if in_years:
            return years
        elif in_months:
            return [(x - dates[0]).days/30.5 for x in dates]
        else:
            return [(x - dates[0]).days for x in dates]

    def get_psa_and_volume(self, patient_no, all=False, last=False):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        psa_density = self.diag_dict[str(patient_no)][2]
        volume = self.diag_dict[str(patient_no)][3]
        if all:
            return psa_density, volume
        elif last:
            return [psa_density[-1]], [volume[-1]]
        else:
            if self.diag_dict[str(patient_no)][-1] == 1:
                return [psa_density[-1]], [volume[-1]]
            else:
                return psa_density, volume

    def get_gg_and_mmcancer(self, patient_no, all=False, last=False):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        gg = [tup[0] for tup in self.diag_dict[str(patient_no)][0]]
        mmc = self.diag_dict[str(patient_no)][1]
        if all:
            return gg, mmc
        elif last:
            return [gg[-1]], [mmc[-1]]
        else:
            if self.diag_dict[str(patient_no)][-1] == 1:
                return [gg[-1]], [mmc[-1]]
            else:
                return gg, mmc

    def get_gleason_diagnosis(self, patient_no):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        return self.diag_dict[str(patient_no)][0]

    def get_gleason_grade_groups(self, patient_no, wsi_name=None):
        assert patient_no in self.patients, f"Patient {patient_no} not found"
        wsi_dict = self.get_patient_data(patient_no)
        if wsi_name is not None:
            assert wsi_name in wsi_dict, f"Slide {wsi_name} not found for patient {patient_no}"
        else:
            wsi_name = list(wsi_dict.keys())[-1]

        idxs = [index for index, wsi_id in enumerate(self.slide_dict[str(patient_no)][0]) if wsi_id.startswith(wsi_name.replace(" ", ""))]
        ggg = [self.slide_dict[str(patient_no)][2][i] for i in idxs]
        ret_dict = {k: v for k, v in zip(wsi_dict[wsi_name], ggg)}

        return ret_dict
        


    @staticmethod
    def check_num(num):
        if type(num) == int or type(num) == float:
            return num
        elif type(num) == str:
            if any(x.isdigit() for x in num):
                num = num.replace(',', '.')
                try:
                    return float(num)
                except ValueError:
                    return np.nan
            else:
                return np.nan
        else:
            return np.nan

    @staticmethod
    def check_date(date):
        if type(date) == str:
            if date[0].isdigit():
                date = datetime.strptime(date, "%d/%m/%Y")
            else:
                date = None
        elif type(date) == datetime:
            str_date = datetime.strftime(date, "%m/%d/%Y")
            date = datetime.strptime(str_date, "%d/%m/%Y")
        else:
            date = None

        return date

    @staticmethod
    def get_gleason_gg(gleason):
        if pd.isna(gleason):
            return 0
        elif gleason == 'B':
            return 0
        elif gleason[0].isdigit() and gleason[2].isdigit():
            g = (int(gleason[0]), int(gleason[2]))
            if sum(g) <= 6:
                grade_group = 1
            elif sum(g) == 7:
                if g[0] == 3:
                    grade_group = 2
                else:
                    grade_group = 3
            elif sum(g) == 8:
                grade_group = 4
            elif sum(g) >= 9:
                grade_group = 5
            else:
                grade_group = 0

            return grade_group
        else:
            return 0



def ispat(p):
    if pd.isna(p):
        return False
    elif not str(p).isnumeric():
        return False
    else:
        return True


def getnum(s):
    if type(s) == str:
        num = re.findall(r"[-]?(?:\d*\.*\d+)", str(s).replace(',', '.'))
        sum = 0
        for n in num:
            sum += float(n)
        return sum
    else:
        return s

def get_slide_name(input):
    yy = input[:2]
    site = input[2:4]
    if not yy.isnumeric():
        yy, site = site, yy
    id_list = input[4:].split()
    if len(id_list) == 1:
        id = id_list[0].split("-")[0]
    else:
        id = id_list[0]

    while len(id) < 5:
        id = f"0{id}"

    return f"{yy}{site} {id}"

def trajectory_plot(dataobj, ps):
    plt.figure(0)
    for p in ps:
        delta = dataobj.get_biopsy_timeline(p)
        y = [p for n in range(len(delta))]
        #delta = [(x - dates[0]).days for x in dates]
        delta = np.round(np.array(delta)/365)
        plt.plot(delta, y, linestyle='solid', linewidth=1, marker="o", markersize=3, markeredgecolor='black')

    plt.show()


def plot_psa_vs_gleason(dataobj, ps):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('PSA/Volume vs Gleason score/mmcancer')
    for p in ps:
        psa, vol = dataobj.get_psa_and_volume(p, all=False)
        gs, mmc = dataobj.get_gg_and_mmcancer(p, all=False)
        label = dataobj.get_patient_label(p)
        ax1.scatter(psa, gs, color='r' if label == 1 else 'g', marker='o', s=10)
        ax2.scatter(vol, mmc, color='r' if label == 1 else 'g', marker='o', s=10)

    ax1.set(xlabel='PSA density', ylabel='Gleason score')
    ax2.set(xlabel='Prostate volume', ylabel='mmcancer')

    plt.show()


if __name__ == "__main__":
    Data = PRIAS_Data_Object("/home/fi5666wi/Documents/PRIAS sheets/20220912_PRIAS log of slides NEW2022 240522 PSA and VOL.xlsx", sheet_name=0)
    #some_patients = Data.get_patient_idxs() #[2, 5, 46, 150, 345]
    #Data.get_gleason_grade_groups(2, '11PM 11268')
    print(Data.get_biopsy_timeline(128))

    #xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_from_Edvard_joined_with_AK.xlsx"
    xcl_path = "/home/fi5666wi/Documents/PRIAS sheets/PRIAS_labels.xlsx"
    some_patients = pd.read_excel(xcl_path, sheet_name=2)['Patient number']
    #plot_psa_vs_gleason(Data, some_patients)
    trajectory_plot(Data, some_patients)

    base_dir = "/home/fi5666wi/PRIAS_data/wsis"
    sum_visits = 0
    isup = []
    for p in some_patients:
        if p in Data.get_patient_idxs():
            #info = Data.info_dict[str(p)]
            #psa, vol = Data.get_psa_and_volume(p)
            #print(f"PSA_{p}: {psa}")
            #print(f"Volume_{p}: {vol}")
            gg_dict = Data.get_gleason_grade_groups(p)
            print(f"GG_{p}: {gg_dict}")
            isup = isup + [v for v in gg_dict.values()]

            wsi_dict = Data.get_patient_data(p)
            label = Data.get_patient_label(p)
            n_visits = len(wsi_dict) if label == 0 else 1
            #last = wsi_dict.popitem()
            for n, wsi in enumerate(wsi_dict):
                # If label is 1 (treated), check only last case
                if label == 1 and n+1 < len(wsi_dict):
                    continue
                wsi_paths = [f"{wsi}-{num}_10x.png" for num in wsi_dict[wsi]]
                for wp in wsi_paths:
                    #print(f"{wp} of {p} exists")
                    if int(wp[:2]) < 11:
                        print(f"Too old slides for {p} ({wp})")
                        #n_visits -= 1
                        break
                    if not os.path.exists(os.path.join(base_dir, wp)):
                        print(f"{wp} does not exist")

            sum_visits += n_visits
            print(f"Patient {p}: {label}, {n_visits} visits")
        else:
            print(f"Patient {p} not in file")

    print(f"Total number of visits: {sum_visits} (Note that some does not exist, should be 53 for test)")

    import matplotlib.pyplot as plt
    #plt.hist(isup, bins=range(7))
    counts, edges, bars = plt.hist(isup, bins=range(7))

    plt.bar_label(bars)
    plt.show()

