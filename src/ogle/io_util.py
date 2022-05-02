from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ogle.constants import DATA_FILE_NAME, DEFAULT_DATA_PATH
from ogle.random_data import generate_parabolic_data


def read_data(data_path=None, is_random=False):
    if is_random:
        return generate_parabolic_data()
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    df = pd.read_csv(data_path)
    return None, df["x"].to_numpy(), df["y"].to_numpy()


def search_data_paths(root_path: Path) -> List[Path]:
    if root_path.is_file():
        return [root_path]
    data_paths = []
    for inner_path in root_path.iterdir():
        if inner_path.is_file() and inner_path.name == f"{DATA_FILE_NAME}.dat":
            data_paths.append(inner_path)
        if inner_path.is_dir():
            data_paths.extend(search_data_paths(inner_path))
    return data_paths


def build_data(data_path: Path):
    with open(data_path) as fd:
        rows = fd.readlines()
    float_rows = [list(map(float, row.replace("\n", "").split())) for row in rows]
    x, y, _, _, _ = zip(*float_rows)
    x, y = np.array(x), np.array(y)
    x -= x[0]
    y = np.power(10, (y[0] - y) / 2.5)
    return x, y
