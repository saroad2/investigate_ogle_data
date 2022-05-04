from typing import Optional

import numpy as np
from ogle.constants import CHI2_EPSILON
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity
from uncertainties import ufloat
from uncertainties.umath import *  # noqa: F403


def iterative_grid_search_2d(
    x,
    y,
    t0_candidate,
    u_min_candidate,
    tau,
    f_bl,
    search_space,
    chi2_epsilon: float = CHI2_EPSILON,
    t0_step: Optional[float] = None,
    u_min_step: Optional[float] = None,
):
    if t0_step is None:
        t0_step = t0_candidate / search_space
    if u_min_step is None:
        u_min_step = u_min_candidate / search_space
    prev_min_chi2 = None
    index = 1
    while True:
        t0_values = [
            t0_candidate + n * t0_step
            for n in range(-search_space // 2, search_space // 2)
        ]
        u_min_values = [
            u_min_candidate + n * u_min_step
            for n in range(-search_space // 2, search_space // 2)
        ]
        chi2_grid = grid_search_2d(
            x, y, t0_values=t0_values, u_min_values=u_min_values, tau=tau, f_bl=f_bl
        )
        result = extract_grid_search_2d_results(
            chi2_grid=chi2_grid, t0_values=t0_values, u_min_values=u_min_values
        )
        min_chi2 = result["chi2"]
        t0_candidate, u_min_candidate = (result["t0"], result["u_min"])
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - min_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
        ):
            result["iterations"] = index
            result["iterations_error"] = 0
            return result
        prev_min_chi2 = min_chi2
        t0_step, u_min_step = t0_step / 2, u_min_step / 2
        index += 1


def grid_search_2d(x, y, t0_values, u_min_values, tau, f_bl):
    n, m = len(t0_values), len(u_min_values)
    chi2_grid = np.zeros(shape=(n, m))
    degrees_of_freedom = x.shape[0] - 2
    for i in range(n):
        for j in range(m):
            t0, u_min = t0_values[i], u_min_values[j]
            intensity = calculate_intensity(t=x, t0=t0, tau=tau, u_min=u_min, f_bl=f_bl)
            chi2_grid[i, j] = calculate_chi2(
                y_true=y, y_pred=intensity, degrees_of_freedom=degrees_of_freedom
            )
    return chi2_grid


def extract_grid_search_2d_results(chi2_grid, t0_values, u_min_values):
    i, j = np.unravel_index(chi2_grid.argmin(), chi2_grid.shape)
    t0_val, u_min_val = t0_values[i], u_min_values[j]
    min_chi2 = float(chi2_grid[i, j])
    t0_error = get_error(t0_values, chi2_grid[:, j])
    u_min_error = get_error(u_min_values, chi2_grid[i, :])
    u_min = ufloat(u_min_val, u_min_error)
    f_max = (u_min**2 + 2) / (u_min * sqrt(u_min**2 + 4))  # noqa: F405
    return dict(
        t0=t0_val,
        t0_error=t0_error,
        u_min=u_min_val,
        u_min_error=u_min_error,
        f_max=f_max.nominal_value,
        f_max_error=f_max.std_dev,
        chi2=min_chi2,
    )


def get_error(values, chi2):
    values = np.array(values)[chi2 < 1]
    return (values.max() - values.min()) / 2
