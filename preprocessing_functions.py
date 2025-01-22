import os
import pathlib
from typing import List, Tuple
from datetime import date
import numpy as np
import pandas as pd


def get_dates(filenames: List[str]) -> List[Tuple[int]]:
    dates_list: List[Tuple[int]] = []
    for filename in filenames:
        if filename.endswith('.csv'):
            filename = filename[:-4]
            filename = filename.split('_')
            try:
                day, month, year = int(filename[-2]), int(filename[-3]), int(filename[-1])
            except:
                day, month, year = int(filename[-3]), int(filename[-4]), int(filename[-2])

            dates_list.append(date(day=day, month=month, year=2000+year))

    # Sort the list according to tuples
    dates_list = sorted(dates_list)

    return dates_list

def create_file_path(battery_dataset_folder, battery_name, date):
    filename = battery_dataset_folder / (battery_name + '_' + '_'.join([str(f'{date.month}'), str(f'{date.day:02d}'), str(date.year-2000)]) + '.csv')
    if not(os.path.isfile(filename)):
        filename = battery_dataset_folder / (battery_name + '_' + '_'.join([str(date.month), str(date.day), str(date.year-2000)]) + '.csv')
    return filename

def get_capacities_and_cycle_index(df):
    cycle_index = df['Cycle_Index'].unique()
    discharge_capacities = np.zeros(len(cycle_index))
    #charge_capacities = np.zeros(len(cycle_index))

    for index in cycle_index:
        if df.loc[df['Cycle_Index'] == index].iloc[-1]['Step_Index'] == 9:
            discharge_capacities[index-1] = df.loc[df['Cycle_Index'] == index, 'Discharge_Capacity(Ah)'].to_list()[-1]
        #charge_capacities[index-1] = df.loc[df['Cycle_Index'] == index, 'Charge_Capacity(Ah)'].to_list()[-1]

    cycle_index = cycle_index[discharge_capacities != 0]
    discharge_capacities = discharge_capacities[discharge_capacities != 0]
    discharge_capacities = np.concat([np.array([discharge_capacities[0]]), np.diff(discharge_capacities)])
    #charge_capacities = np.concat([np.array([charge_capacities[0]]), np.diff(charge_capacities)])

    df_dict = {
        'Cycle_Index': cycle_index,
        'Discharge_Capacity(Ah)': discharge_capacities
    }
    return pd.DataFrame(df_dict)

def process_battery_dataset(battery_folder_path, battery_name, dates):
    fin_df = None
    for i, date in enumerate(dates):
        file_path = create_file_path(battery_folder_path, battery_name, date)
        df = pd.read_csv(file_path)
        extracted_capacities = get_capacities_and_cycle_index(df)
        if i == 0:
            fin_df = extracted_capacities
            #fin_df['File_Path'] = str(file_path)
            #fin_df['File_Cycle_Index'] = fin_df['Cycle_Index']
        else:
            extracted_capacities.index += (fin_df.index[-1]+1)
            #extracted_capacities['File_Path'] = str(file_path)
            #extracted_capacities['File_Cycle_Index'] = extracted_capacities['Cycle_Index']
            extracted_capacities['Cycle_Index'] += fin_df.loc[fin_df.index[-1], 'Cycle_Index']
            fin_df = pd.concat([fin_df, extracted_capacities])
    return fin_df

def drop_outlier_discharge_cycles(df,bins):
    index = []
    discharge_capacities = df['Discharge_Capacity(Ah)']
    range_ = np.arange(1,df.shape[0],bins)
    for i in range_[:-1]:
        array_lim = discharge_capacities[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.median(array_lim)
        th_max,th_min = mean + sigma*1.2, mean - sigma*1
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return df.iloc[index, :]

def generate_sliding_window_data(sequence, window_size=10):
    x, y = [],[]
    for i in range(len(sequence) - window_size):
        features = sequence[i:i+window_size]
        target = sequence[i+window_size]

        x.append(features)
        y.append(target)
        
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)

