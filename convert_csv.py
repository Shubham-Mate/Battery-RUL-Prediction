import pandas as pd
from constants import CALCEDatasetConstants, CustomDatasetConstants
import preprocessing_functions
import os



for i, path in enumerate(CALCEDatasetConstants.BATTERY_DATASET_PATHS.value):
    for file in os.listdir(str(path)):
        df = pd.read_excel(path / file, engine='calamine', sheet_name=1)
        df.to_csv(CALCEDatasetConstants.BATTERY_DATASET_PATHS_CSV.value[i] / (file.split('.')[0]+'.csv'))
        print(f"Converted {file}")


xlsx_path = os.path.splitext(str(CustomDatasetConstants.BATTERY_RAW_DATASET_PATHS.value[0]))[0] + '.xlsx'
df = pd.read_excel(xlsx_path, engine='calamine', sheet_name=0)
df.to_csv(CustomDatasetConstants.BATTERY_RAW_DATASET_PATHS.value[0])
print(f"Converted {CustomDatasetConstants.BATTERY_RAW_DATASET_PATHS.value[0]}")