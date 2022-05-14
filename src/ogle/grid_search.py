import itertools
from typing import Optional

import numpy as np
import pandas as pd
from ogle.constants import CHI2_EPSILON
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity
from ogle.random_data import sample_records
from ogle.search_point import SearchPoint


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
    max_iterations: Optional[int] = None,
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
        best_approximation, best_chi2 = extract_grid_search_best_approximation(
            chi2_grid_table
        )
        candidates_dict = {
            key: value
            for key, value in best_approximation.asdict().items()
            if key not in constants_dict.keys()
        }
        if verbose:
            print(f"Min chi2: {best_chi2:.2e}")
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - best_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
            or index == max_iterations  # noqa: W503
        ):
            break
        prev_min_chi2 = best_chi2
        steps_dict = {key: value / 2 for key, value in steps_dict.items()}
        index += 1
    if history:
        return chi2_grid_tables_history
    last_grid_table = chi2_grid_tables_history[-1]
    return build_results_dict(
        chi2_grid_table=last_grid_table,
        parameters=parameters,
        index=index,
        x=x,
        y=y,
        yerr=yerr,
        steps_dict=steps_dict,
    )


def build_results_dict(
    chi2_grid_table,
    parameters,
    index,
    x,
    y,
    yerr,
    steps_dict,
):
    best_approximation, best_chi2 = extract_grid_search_best_approximation(
        chi2_grid_table
    )
    results = best_approximation.asdict()
    results.update(
        calculate_errors_dict(
            best_approximation=best_approximation,
            best_chi2=best_chi2,
            parameters=parameters,
            steps_dict=steps_dict,
            x=x,
            y=y,
            yerr=yerr,
        )
    )
    results["iterations"] = index
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
    best_chi2 = chi2_grid_table["chi2"].min()
    min_chi2_records = chi2_grid_table.loc[
        chi2_grid_table["chi2"] == best_chi2
    ].to_dict("records")
    if len(min_chi2_records) == 0:
        return None
    best_record = min_chi2_records[0]
    best_record.pop("chi2")
    return SearchPoint(**best_record), best_chi2


def build_grid_matrix(chi2_grid_table, best_approximation, parameter1, parameter2):
    filter_dict = best_approximation.asdict()
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


def calculate_errors_dict(
    best_approximation, best_chi2, parameters, steps_dict, x, y, yerr
):
    errors_dict = {}
    for parameter in parameters:
        errors_dict[f"{parameter}_error"] = get_error(
            best_approximation=best_approximation,
            best_chi2=best_chi2,
            parameter=parameter,
            step=steps_dict[parameter],
            x=x,
            y=y,
            yerr=yerr,
            max_search=1_000,
        )

    return errors_dict


def get_error(best_approximation, best_chi2, parameter, step, x, y, yerr, max_search):
    max_value = find_value(
        best_approximation=best_approximation,
        best_chi2=best_chi2,
        parameter=parameter,
        step=step,
        x=x,
        y=y,
        yerr=yerr,
        direction=1,
        max_search=max_search,
    )
    min_value = find_value(
        best_approximation=best_approximation,
        best_chi2=best_chi2,
        parameter=parameter,
        step=step,
        x=x,
        y=y,
        yerr=yerr,
        direction=-1,
        max_search=max_search,
    )
    if max_value is None or min_value is None:
        return None
    return (max_value - min_value) / 2


def find_value(
    best_approximation, best_chi2, parameter, step, x, y, yerr, direction, max_search
):
    degrees_of_freedom = x.shape[0] - 2
    best_parameter_value = best_approximation.asdict()[parameter]
    for i in range(1, max_search):
        parameter_value = best_parameter_value + direction * i * step
        search_point = best_approximation + SearchPoint(**{parameter: parameter_value})
        intensity = calculate_intensity(t=x, **search_point.asdict())
        chi2 = calculate_chi2(
            y_true=y, y_pred=intensity, yerr=yerr, degrees_of_freedom=degrees_of_freedom
        )
        if chi2 > best_chi2 + 1:
            return parameter_value
    return None


def build_values_dict(chi2_grid_table, parameters):
    values_dict = {}
    for parameter in parameters:
        values = list(chi2_grid_table[parameter].unique())
        values.sort()
        values_dict[parameter] = values
    return values_dict
