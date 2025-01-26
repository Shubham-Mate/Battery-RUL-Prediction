from enum import Enum
import pathlib


class CALCEDatasetConstants(Enum):
    BATTERY_NUMBERS = [35, 36, 37, 38]
    BATTERY_NAMES = ['CS2_' + str(i) for i in BATTERY_NUMBERS]
    BATTERY_DATASET_PATHS = [pathlib.Path(__file__).parent / "data" / "CALCE" / 'Raw' / name for name in BATTERY_NAMES]
    BATTERY_PROCESSED_PATH = pathlib.Path(__file__).parent / "data" / "CALCE" / 'Preprocessed'
    BATTERY_DATASET_PATHS_CSV = [pathlib.Path(__file__).parent / "data" / "CALCE" / 'CSV' / name for name in BATTERY_NAMES]

MODEL_OUTPUT_PATH = pathlib.Path(__file__).parent / 'saved models'
