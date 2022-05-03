import numpy as np
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity, calculate_mu


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
    t0, u_min = t0_values[i], u_min_values[j]
    min_chi2 = float(chi2_grid[i, j])
    t0_error = get_error(t0_values, chi2_grid[:, j])
    u_min_error = get_error(u_min_values, chi2_grid[i, :])
    return dict(
        t0=t0,
        t0_error=t0_error,
        u_min=u_min,
        u_min_error=u_min_error,
        f_max=calculate_mu(u_min),
        chi2=min_chi2,
    )


def get_error(values, chi2):
    values = np.array(values)[chi2 < 1]
    return (values.max() - values.min()) / 2
