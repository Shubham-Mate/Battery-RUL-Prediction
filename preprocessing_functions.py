import os
import pathlib
from typing import List, Tuple
from datetime import date


def get_dates(filenames: List[str]) -> List[Tuple[int]]:
    dates_list: List[Tuple[int]] = []
    for filename in filenames:
        if filename.endswith('.xlsx') or filename.endswith('.txt'):
            filename = filename[:-5]
            filename = filename.split('_')
            try:
                day, month, year = int(filename[-2]), int(filename[-3]), int(filename[-1])
            except:
                day, month, year = int(filename[-3]), int(filename[-4]), int(filename[-2])

            dates_list.append(date(day=day, month=month, year=2000+year))

    # Sort the list according to tuples
    dates_list = sorted(dates_list)

    return dates_list
