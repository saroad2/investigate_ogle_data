import json
from pathlib import Path

import click
import numpy as np
from joblib import Parallel, delayed
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.constants import (
    CHI2_EPSILON,
    DATA_FILE_NAME,
    DEFAULT_EXPERIMENTS,
    DEFAULT_SPACE_SEARCH,
    GRID_SEARCH_2D_NAMES,
    GRID_SEARCH_4D_NAMES,
)
from ogle.grid_search import iterative_grid_search
from ogle.io_util import read_data
from ogle.plot_util import plot_grid_search_results, plot_monte_carlo_results


@ogle_cli_group.group("grid-search")
def grid_search_cli_group():
    """Grid-search related commands."""


@grid_search_cli_group.command("2d-search")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("--tau", type=float, required=True)
@click.option("--fbl", type=float, default=1)
@click.option("--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("--chi2-epsilon", type=float, default=CHI2_EPSILON)
@click.option("-m", "--max-iterations", type=int)
@click.option("--limited/--no-limited", is_flag=True, default=False)
def grid_search_2d_cli(
    data_path, tau, fbl, search_space, chi2_epsilon, max_iterations, limited
):
    if data_path.is_dir():
        data_path = data_path / f"{DATA_FILE_NAME}.csv"
    data_dir = data_path.parent
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / "parabolic_fitting_results" / "fit_result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    output_dir = data_dir / "grid_search_2d"
    output_dir.mkdir(exist_ok=True)
    history = iterative_grid_search(
        x=x,
        y=y,
        yerr=yerr,
        constants_dict=dict(tau=tau, f_bl=fbl),
        candidates_dict=dict(t0=t0_candidate, u_min=u_min_candidate),
        steps_dict=dict(t0=t0_step, u_min=u_min_step),
        search_space=search_space,
        chi2_epsilon=chi2_epsilon,
        limited=limited,
        verbose=True,
        max_iterations=max_iterations,
    )
    for i, chi2_grid_table in enumerate(history, start=1):
        plot_grid_search_results(
            x=x,
            y=y,
            yerr=yerr,
            chi2_grid_table=chi2_grid_table,
            parameters=["t0", "u_min"],
            output_dir=output_dir,
            index=i,
            steps_dict=dict(
                t0=t0_step / np.power(2, i), u_min=u_min_step / np.power(2, i)
            ),
        )


@grid_search_cli_group.command("2d-monte-carlo")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("--tau", type=float, required=True)
@click.option("--fbl", type=float, default=1)
@click.option("--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
@click.option("--normal-curve/--no-normal-curve", is_flag=True, default=True)
@click.option("--chi2-epsilon", type=float, default=CHI2_EPSILON)
@click.option("-w", "--workers", type=int, default=1)
@click.option("-m", "--max-iterations", type=int)
def monte_carlo_2d_cli(
    data_path,
    tau,
    fbl,
    search_space,
    experiments,
    chi2_epsilon,
    normal_curve,
    workers,
    max_iterations,
):
    if data_path.is_dir():
        data_path = data_path / f"{DATA_FILE_NAME}.csv"
    data_dir = data_path.parent
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / "parabolic_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    results = Parallel(n_jobs=workers, verbose=5)(
        delayed(iterative_grid_search)(
            x=x,
            y=y,
            yerr=yerr,
            constants_dict=dict(tau=tau, f_bl=fbl),
            candidates_dict=dict(t0=t0_candidate, u_min=u_min_candidate),
            steps_dict=dict(t0=t0_step, u_min=u_min_step),
            search_space=search_space,
            chi2_epsilon=chi2_epsilon,
            sample=True,
            history=False,
            max_iterations=max_iterations,
        )
        for _ in range(experiments)
    )
    plot_monte_carlo_results(
        results,
        parameters=GRID_SEARCH_2D_NAMES + ["iterations"],
        output_dir=data_dir / "grid_search_2d_monte_carlo_results",
        normal_curve=normal_curve,
    )


@grid_search_cli_group.command("4d-search")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("-s", "--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("-c", "--chi2-epsilon", type=float, default=CHI2_EPSILON)
@click.option("-m", "--max-iterations", type=int)
@click.option("--limited/--no-limited", is_flag=True, default=False)
def grid_search_4d_cli(data_path, search_space, chi2_epsilon, max_iterations, limited):
    if data_path.is_dir():
        data_path = data_path / f"{DATA_FILE_NAME}.csv"
    data_dir = data_path.parent
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / "parabolic_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    tau_candidate, tau_step = results_json["tau"], results_json["tau_error"]
    fbl_candidate, fbl_step = 1, 2 / search_space
    output_dir = data_dir / "grid_search_4d"
    output_dir.mkdir(exist_ok=True)
    history = iterative_grid_search(
        x=x,
        y=y,
        yerr=yerr,
        constants_dict={},
        candidates_dict=dict(
            t0=t0_candidate,
            tau=tau_candidate,
            u_min=u_min_candidate,
            f_bl=fbl_candidate,
        ),
        steps_dict=dict(t0=t0_step, tau=tau_step, u_min=u_min_step, f_bl=fbl_step),
        search_space=search_space,
        chi2_epsilon=chi2_epsilon,
        verbose=True,
        limited=limited,
        max_iterations=max_iterations,
    )
    for i, chi2_grid_table in enumerate(history, start=1):
        plot_grid_search_results(
            x=x,
            y=y,
            yerr=yerr,
            chi2_grid_table=chi2_grid_table,
            parameters=["t0", "u_min", "tau", "f_bl"],
            output_dir=output_dir,
            index=i,
            steps_dict=dict(
                t0=t0_step / np.power(2, i),
                u_min=u_min_step / np.power(2, i),
                tau=tau_step / np.power(2, i),
                f_bl=fbl_step / np.power(2, i),
            ),
        )


@grid_search_cli_group.command("4d-monte-carlo")
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
@click.option("-s", "--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
@click.option("--normal-curve/--no-normal-curve", is_flag=True, default=True)
@click.option("-c", "--chi2-epsilon", type=float, default=CHI2_EPSILON)
@click.option("-w", "--workers", type=int, default=1)
@click.option("-m", "--max-iterations", type=int)
def monte_carlo_4d_cli(
    data_path,
    search_space,
    experiments,
    chi2_epsilon,
    normal_curve,
    workers,
    max_iterations,
):
    if data_path.is_dir():
        data_path = data_path / f"{DATA_FILE_NAME}.csv"
    data_dir = data_path.parent
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / "parabolic_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    tau_candidate, tau_step = results_json["tau"], results_json["tau_error"]
    fbl_candidate, fbl_step = 0.5, 2 / search_space
    results = Parallel(n_jobs=workers, verbose=5)(
        delayed(iterative_grid_search)(
            x=x,
            y=y,
            yerr=yerr,
            constants_dict={},
            candidates_dict=dict(
                t0=t0_candidate,
                tau=tau_candidate,
                u_min=u_min_candidate,
                f_bl=fbl_candidate,
            ),
            steps_dict=dict(t0=t0_step, tau=tau_step, u_min=u_min_step, f_bl=fbl_step),
            search_space=search_space,
            chi2_epsilon=chi2_epsilon,
            sample=True,
            history=False,
            max_iterations=max_iterations,
        )
        for _ in range(experiments)
    )
    plot_monte_carlo_results(
        results,
        parameters=GRID_SEARCH_4D_NAMES + ["iterations"],
        output_dir=data_dir / "grid_search_4d_monte_carlo_results",
        normal_curve=normal_curve,
    )
