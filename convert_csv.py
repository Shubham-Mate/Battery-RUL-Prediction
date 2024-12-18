import pandas as pd
from constants import CALCEDatasetConstants
import preprocessing_functions
import os



for i, path in enumerate(CALCEDatasetConstants.BATTERY_DATASET_PATHS.value):
    for file in os.listdir(str(path)):
        df = pd.read_excel(path / file, engine='calamine', sheet_name=1)
        df.to_csv(CALCEDatasetConstants.BATTERY_DATASET_PATHS_CSV.value[i] / (file.split('.')[0]+'.csv'))
        print(f"Converted {file}")