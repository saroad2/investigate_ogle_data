import itertools
from typing import Optional

import numpy as np
import pandas as pd
from ogle.constants import CHI2_EPSILON
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity
from ogle.random_data import sample_records


def iterative_grid_search_2d(
    x,
    y,
    yerr,
    t0_candidate,
    u_min_candidate,
    tau,
    f_bl,
    search_space,
    chi2_epsilon: float = CHI2_EPSILON,
    t0_step: Optional[float] = None,
    u_min_step: Optional[float] = None,
    sample: bool = False,
):
    if t0_step is None:
        t0_step = t0_candidate / search_space
    if u_min_step is None:
        u_min_step = u_min_candidate / search_space
    if sample:
        x, y, yerr = sample_records(x, y, yerr)
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
        chi2_grid_table = grid_search(
            x, y, yerr, t0=t0_values, u_min=u_min_values, tau=[tau], f_bl=[f_bl]
        )
        best_approximation = extract_grid_search_best_approximation(chi2_grid_table)
        best_chi2 = best_approximation["chi2"]
        results = dict(best_approximation)
        results.update(
            calculate_errors_dict(
                chi2_grid_table, best_approximation, best_chi2=best_chi2
            )
        )
        t0_candidate, u_min_candidate = (results["t0"], results["u_min"])
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - best_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
        ):
            results["iterations"] = index
            results["iterations_error"] = 0
            return results
        prev_min_chi2 = best_chi2
        t0_step, u_min_step = t0_step / 2, u_min_step / 2
        index += 1


def grid_search(x, y, yerr, **kwargs):
    parameters = list(kwargs.keys())
    indexes_permutations = itertools.product(
        *[range(len(values)) for values in kwargs.values()]
    )
    columns = list(parameters)
    columns.append("chi2")
    rows = []
    degrees_of_freedom = x.shape[0] - 2
    for k, indexes_permutation in enumerate(indexes_permutations):
        row = dict()
        parameters_dict = {
            parameters[i]: kwargs[parameters[i]][j]
            for i, j in enumerate(indexes_permutation)
        }
        row.update(parameters_dict)
        intensity = calculate_intensity(t=x, **parameters_dict)
        row["chi2"] = calculate_chi2(
            y_true=y, y_pred=intensity, yerr=yerr, degrees_of_freedom=degrees_of_freedom
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def extract_grid_search_best_approximation(chi2_grid_table):
    return chi2_grid_table[
        chi2_grid_table["chi2"] == chi2_grid_table["chi2"].min()
    ].to_dict("records")[0]


def build_grid_matrix(chi2_grid_table, best_approximation, parameter1, parameter2):
    filter_dict = dict(best_approximation)
    del filter_dict[parameter1]
    del filter_dict[parameter2]
    filter_dict = {
        key: value for key, value in filter_dict.items() if not key.endswith("index")
    }
    filtered_table = chi2_grid_table.loc[
        (chi2_grid_table[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)
    ]
    parameter1_values = list(filtered_table[parameter1].unique())
    parameter1_values.sort()
    parameter2_values = list(filtered_table[parameter2].unique())
    parameter2_values.sort()
    n = len(parameter1_values)
    m = len(parameter2_values)
    grid = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            parameter1_value, parameter2_value = (
                parameter1_values[i],
                parameter2_values[j],
            )
            record = filtered_table[
                filtered_table[parameter1].eq(parameter1_value)
                & filtered_table[parameter2].eq(parameter2_value)  # noqa: W503
            ]
            grid[i, j] = record["chi2"].iloc[0]
    return grid


def calculate_errors_dict(chi2_grid_table, best_approximation, best_chi2):
    errors_dict = {}
    for parameter in best_approximation.keys():
        filter_dict = dict(best_approximation)
        del filter_dict[parameter]
        filtered_table = chi2_grid_table.loc[
            (chi2_grid_table[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)
        ]
        parameter_values = filtered_table[parameter].to_numpy()
        chi2_values = filtered_table["chi2"].to_numpy()
        errors_dict[f"{parameter}_error"] = get_error(
            values=parameter_values,
            chi2=chi2_values,
            best_chi2=best_chi2,
            best_parameter=best_approximation[parameter],
        )

    return errors_dict


def get_error(values, chi2, best_chi2, best_parameter):
    filtered_values = values[np.fabs(chi2 - best_chi2) > 1]
    if filtered_values.shape[0] == 0:
        return None
    max_value = np.min(filtered_values[filtered_values > best_parameter])
    min_value = np.max(filtered_values[filtered_values < best_parameter])
    return (max_value - min_value) / 2
