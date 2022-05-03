import numpy as np
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity


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
