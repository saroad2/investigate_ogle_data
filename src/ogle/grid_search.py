import itertools

import numpy as np
import pandas as pd
from ogle.constants import CHI2_EPSILON
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity
from ogle.random_data import sample_records


def create_search_list(candidate, step, search_space):
    return [candidate + n * step for n in range(-search_space // 2, search_space // 2)]


def iterative_grid_search(
    x,
    y,
    yerr,
    constants_dict,
    candidates_dict,
    steps_dict,
    search_space,
    chi2_epsilon: float = CHI2_EPSILON,
    sample: bool = False,
    verbose: bool = False,
    history: bool = True,
):
    for key, value in candidates_dict.items():
        if key not in steps_dict:
            steps_dict[key] = value / search_space
    if sample:
        x, y, yerr = sample_records(x, y, yerr)
    prev_min_chi2 = None
    index = 1
    parameters = list(candidates_dict.keys())
    chi2_grid_tables_history = []
    while True:
        if verbose:
            print(f"Grid search {index}")
        values_dict = {key: [value] for key, value in constants_dict.items()}
        values_dict.update(
            {
                key: create_search_list(value, steps_dict[key], search_space)
                for key, value in candidates_dict.items()
            }
        )
        chi2_grid_table = grid_search(x, y, yerr, **values_dict)
        chi2_grid_tables_history.append(chi2_grid_table)
        best_approximation = extract_grid_search_best_approximation(chi2_grid_table)
        best_chi2 = best_approximation.pop("chi2")
        candidates_dict = {
            key: value
            for key, value in best_approximation.items()
            if key not in constants_dict
        }
        if verbose:
            print(f"Min chi2: {best_chi2:.2e}")
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - best_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
        ):
            break
        prev_min_chi2 = best_chi2
        steps_dict = {key: value / 2 for key, value in steps_dict.items()}
        index += 1
    if history:
        return chi2_grid_tables_history
    last_grid_table = chi2_grid_tables_history[-1]
    return build_results_dict(
        chi2_grid_table=last_grid_table, parameters=parameters, index=index
    )


def build_results_dict(chi2_grid_table, parameters, index):
    best_approximation = extract_grid_search_best_approximation(chi2_grid_table)
    best_chi2 = best_approximation.pop("chi2")
    results = dict(best_approximation)
    results.update(
        calculate_errors_dict(
            chi2_grid_table,
            best_approximation,
            best_chi2=best_chi2,
            parameters=parameters,
        )
    )
    results["iteration"] = index
    return results


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


def calculate_errors_dict(chi2_grid_table, best_approximation, best_chi2, parameters):
    errors_dict = {}
    for parameter in parameters:
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
        return np.inf
    big_values = filtered_values[filtered_values > best_parameter]
    max_value = np.min(big_values) if big_values.shape[0] > 0 else np.max(values)
    small_values = filtered_values[filtered_values < best_parameter]
    min_value = np.max(small_values) if small_values.shape[0] > 0 else np.min(values)
    return (max_value - min_value) / 2


def build_values_dict(chi2_grid_table, parameters):
    values_dict = {}
    for parameter in parameters:
        values = list(chi2_grid_table[parameter].unique())
        values.sort()
        values_dict[parameter] = values
    return values_dict
