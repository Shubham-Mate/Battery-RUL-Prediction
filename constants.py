from enum import Enum
import pathlib


class CALCEDatasetConstants(Enum):
    BATTERY_NUMBERS = [3, 5, 6, 7, 8, 9, 21, 24, 25, 33, 34, 35, 36, 37, 38]
    BATTERY_NAMES = ['CS2_' + str(i) for i in BATTERY_NUMBERS]
    BATTERY_DATASET_PATHS = [pathlib.Path(__file__).parent / "data" / "CALCE" / name for name in BATTERY_NAMES]
