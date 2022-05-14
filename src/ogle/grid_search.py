import itertools
from typing import Optional

import numpy as np
import pandas as pd
from ogle.constants import CHI2_EPSILON, PARAMETER_TO_LIMITS
from ogle.fit_data import calculate_chi2
from ogle.ogle_util import calculate_intensity
from ogle.random_data import sample_records
from ogle.search_point import SearchPoint


def create_search_list(candidate, step, search_space, parameter, limited: bool = False):
    min_val, max_val = (
        candidate - step * search_space / 2,
        candidate + step * search_space / 2,
    )
    if limited:
        absolute_min, absolute_max = PARAMETER_TO_LIMITS[parameter]
        if absolute_min is not None:
            min_val = max(min_val, absolute_min)
        if absolute_max is not None:
            max_val = min(max_val, absolute_max)
    return np.linspace(min_val, max_val, num=search_space)


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
    limited: bool = False,
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
                key: create_search_list(
                    value, steps_dict[key], search_space, parameter=key, limited=limited
                )
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
    results["chi2"] = best_chi2
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


def build_grid_matrix(
    best_approximation: SearchPoint,
    best_chi2: float,
    parameter1: str,
    parameter2: str,
    step1: float,
    step2: float,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    degrees_of_freedom: int,
    size: int = 50,
):
    parameter1_values = get_parameter_values(
        start_point=best_approximation,
        parameter=parameter1,
        step=step1,
        search_value=best_chi2 + 12,
        x=x,
        y=y,
        yerr=yerr,
        degrees_of_freedom=degrees_of_freedom,
        size=size,
    )
    parameter2_values = get_parameter_values(
        start_point=best_approximation,
        parameter=parameter2,
        step=step2,
        search_value=best_chi2 + 12,
        x=x,
        y=y,
        yerr=yerr,
        degrees_of_freedom=degrees_of_freedom,
        size=size,
    )
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            search_point = best_approximation.replace(
                **{parameter1: parameter1_values[i], parameter2: parameter2_values[j]}
            )
            intensity = calculate_intensity(t=x, **search_point.asdict())
            grid[i, j] = calculate_chi2(
                y_true=y,
                y_pred=intensity,
                yerr=yerr,
                degrees_of_freedom=degrees_of_freedom,
            )
    return parameter1_values, parameter2_values, grid


def get_parameter_values(
    start_point: SearchPoint,
    parameter: str,
    step: float,
    search_value: float,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    degrees_of_freedom: int,
    size: int,
):
    i = 1
    while True:
        search_point = start_point.move(**{parameter: i * step})
        intensity = calculate_intensity(t=x, **search_point.asdict())
        chi2 = calculate_chi2(
            y_true=y, y_pred=intensity, yerr=yerr, degrees_of_freedom=degrees_of_freedom
        )
        if chi2 > search_value:
            max_value = search_point.asdict()[parameter]
            break
        i += 1
    i = -1
    while True:
        search_point = start_point.move(**{parameter: i * step})
        intensity = calculate_intensity(t=x, **search_point.asdict())
        chi2 = calculate_chi2(
            y_true=y, y_pred=intensity, yerr=yerr, degrees_of_freedom=degrees_of_freedom
        )
        if chi2 > search_value:
            min_value = search_point.asdict()[parameter]
            break
        i -= 1
    return np.linspace(min_value, max_value, num=size)
