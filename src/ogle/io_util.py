import pandas as pd
from ogle.constants import DEFAULT_DATA_PATH
from ogle.random_data import generate_parabolic_data


def read_data(data_path=None, is_random=False):
    if is_random:
        return generate_parabolic_data()
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    df = pd.read_csv(data_path)
    return None, df["x"].to_numpy(), df["y"].to_numpy()
