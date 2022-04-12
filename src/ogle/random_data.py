import numpy as np
from ogle.constants import (
    DEFAULT_DATA_POINTS,
    DEFAULT_RELATIVE_ERROR,
    MAX_ARG,
    MAX_X,
    MIN_ARG,
    MIN_X,
)


def generate_parabolic_data(
    data_points=DEFAULT_DATA_POINTS,
    x_rel_err=DEFAULT_RELATIVE_ERROR,
    y_rel_err=DEFAULT_RELATIVE_ERROR,
    min_x=MIN_X,
    max_x=MAX_X,
    min_arg=MIN_ARG,
    max_arg=MAX_ARG,
):
    x = np.random.uniform(min_x, max_x, size=data_points)
    x.sort()
    a = np.random.uniform(min_arg, max_arg, size=3)
    xerr = np.random.normal(loc=1, scale=x_rel_err, size=data_points)
    yerr = np.random.normal(loc=1, scale=y_rel_err, size=data_points)
    y = np.polyval(a, x * xerr) * yerr
    return a, x, y
