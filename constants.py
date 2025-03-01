from enum import Enum
import pathlib


class CALCEDatasetConstants(Enum):
    BATTERY_NUMBERS = [35, 36, 37, 38]
    BATTERY_NAMES = ['CS2_' + str(i) for i in BATTERY_NUMBERS]
    BATTERY_DATASET_PATHS = [pathlib.Path(__file__).parent / "data" / "CALCE" / 'Raw' / name for name in BATTERY_NAMES]
    BATTERY_PROCESSED_PATH = pathlib.Path(__file__).parent / "data" / "CALCE" / 'Preprocessed'
    BATTERY_DATASET_PATHS_CSV = [pathlib.Path(__file__).parent / "data" / "CALCE" / 'CSV' / name for name in BATTERY_NAMES]

class NASADatasetConstants(Enum):
    BATTERY_NAMES = ['B00' + num for num in ['05', '06', '07', '18']]
    BATTERY_FILE_NAMES = [name + '_discharge.csv' for name in BATTERY_NAMES]
    BATTERY_FILEPATHS = [pathlib.Path(__file__).parent / "data" / 'NASA' / file_name for file_name in BATTERY_FILE_NAMES]

class ModelOutputConstants(Enum):
    MODEL_OUTPUT_PATH = pathlib.Path(__file__).parent / 'saved models'
    PLOTS_OUTPUT_PATH = pathlib.Path(__file__).parent / 'plots'
    CALCE_OUTPUT_PATH = MODEL_OUTPUT_PATH / "CALCE"
    NASA_OUTPUT_PATH = MODEL_OUTPUT_PATH / "NASA"
