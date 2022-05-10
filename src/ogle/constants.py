from pathlib import Path

DATA_FILE_NAME = "phot"

DEFAULT_DATA_PATH = Path.home() / "parabolic_data.csv"
DEFAULT_EXPERIMENTS = 10_000
DEFAULT_RELATIVE_ERROR = 0.05
DEFAULT_DATA_POINTS = 100
DEFAULT_SPACE_SEARCH = 50
MIN_X, MAX_X = 0, 10
MIN_ARG, MAX_ARG = -5, 5
CHI2_EPSILON = 1e-5

GRID_SEARCH_4D_NAMES = ["t0", "u_min", "tau", "f_bl"]
GRID_SEARCH_2D_NAMES = ["t0", "u_min"]
MICROLENSING_PROPERTY_NAMES = ["t0", "f_max", "u_min", "tau"]
