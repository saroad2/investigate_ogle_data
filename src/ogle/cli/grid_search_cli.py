import itertools
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
from ogle.grid_search import (
    build_grid_matrix,
    calculate_errors_dict,
    create_search_list,
    extract_grid_search_best_approximation,
    grid_search,
    iterative_grid_search_2d,
)
from ogle.io_util import read_data
from ogle.ogle_util import calculate_intensity
from ogle.plot_util import plot_fit, plot_grid, plot_monte_carlo_results


@ogle_cli_group.group("grid-search")
def grid_search_cli_group():
    """Grid-search related commands."""


@grid_search_cli_group.command("2d-search")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--tau", type=float, required=True)
@click.option("--fbl", type=float, default=1)
@click.option("--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("--chi2-epsilon", type=float, default=CHI2_EPSILON)
def grid_search_2d_cli(data_dir, tau, fbl, search_space, chi2_epsilon):
    data_path = data_dir / f"{DATA_FILE_NAME}.csv"
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / f"{DATA_FILE_NAME}_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    index = 1
    prev_min_chi2 = None
    while True:
        click.echo(f"Grid search {index}")
        t0_values = create_search_list(t0_candidate, t0_step, search_space)
        u_min_values = create_search_list(u_min_candidate, u_min_step, search_space)
        chi2_grid_table = grid_search(
            x=x, y=y, yerr=yerr, t0=t0_values, u_min=u_min_values, tau=[tau], f_bl=[fbl]
        )
        best_approximation = extract_grid_search_best_approximation(
            chi2_grid_table=chi2_grid_table
        )
        best_chi2 = best_approximation.pop("chi2")
        chi2_grid = build_grid_matrix(
            chi2_grid_table, best_approximation, "t0", "u_min"
        )
        output_dir = data_dir / "grid_search_2d"
        output_dir.mkdir(exist_ok=True)
        plot_grid(
            chi2_grid=chi2_grid,
            x_values=t0_values,
            y_values=u_min_values,
            x_parameter="t0",
            y_parameter="u_min",
            output_path=output_dir / f"grid_search{index}.png",
        )
        results = dict(best_approximation)
        results.update(
            calculate_errors_dict(
                chi2_grid_table, best_approximation, best_chi2=best_chi2
            )
        )
        results["chi2"] = best_chi2
        with open(output_dir / f"grid_search{index}_results.json", mode="w") as fd:
            json.dump(results, fd, indent=2)
        plot_fit(
            x=x,
            y_true=y,
            y_pred=calculate_intensity(t=x, **best_approximation),
            yerr=yerr,
            title=rf"Grid search fit ($\chi^2_{{red}} = {best_chi2:.2e}$)",
            xlabel="Time [sec]",
            ylabel="Intensity factor",
            output_file=output_dir / f"grid_search_fit{index}.png",
        )
        t0_candidate, u_min_candidate = results["t0"], results["u_min"]
        click.echo(f"Min chi2: {best_chi2:.2e}")
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - best_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
        ):
            break
        prev_min_chi2 = best_chi2
        t0_step, u_min_step = t0_step / 2, u_min_step / 2
        index += 1


@grid_search_cli_group.command("2d-monte-carlo")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--tau", type=float, required=True)
@click.option("--fbl", type=float, default=1)
@click.option("--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("-e", "--experiments", type=int, default=DEFAULT_EXPERIMENTS)
@click.option("--normal-curve/--no-normal-curve", is_flag=True, default=True)
@click.option("--chi2-epsilon", type=float, default=CHI2_EPSILON)
@click.option("-w", "--workers", type=int, default=1)
def monte_carlo_2d_cli(
    data_dir, tau, fbl, search_space, experiments, chi2_epsilon, normal_curve, workers
):
    data_path = data_dir / f"{DATA_FILE_NAME}.csv"
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / f"{DATA_FILE_NAME}_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate = results_json["t0"]
    u_min_candidate = results_json["u_min"]
    results = Parallel(n_jobs=workers, verbose=5)(
        delayed(iterative_grid_search_2d)(
            x=x,
            y=y,
            yerr=yerr,
            t0_candidate=t0_candidate,
            u_min_candidate=u_min_candidate,
            tau=tau,
            f_bl=fbl,
            search_space=search_space,
            chi2_epsilon=chi2_epsilon,
            sample=True,
        )
        for _ in range(experiments)
    )
    plot_monte_carlo_results(
        results,
        property_names=GRID_SEARCH_2D_NAMES + ["iterations"],
        output_dir=data_dir / "grid_search_2d_monte_carlo_results",
        normal_curve=normal_curve,
    )


@grid_search_cli_group.command("4d-search")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("-s", "--search-space", type=int, default=DEFAULT_SPACE_SEARCH)
@click.option("-c", "--chi2-epsilon", type=float, default=CHI2_EPSILON)
def grid_search_4d_cli(data_dir, search_space, chi2_epsilon):
    data_path = data_dir / f"{DATA_FILE_NAME}.csv"
    _, x, y, yerr = read_data(data_path=data_path, is_random=False)
    with (data_dir / f"{DATA_FILE_NAME}_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"], results_json["t0_error"]
    u_min_candidate, u_min_step = results_json["u_min"], results_json["u_min_error"]
    tau_candidate, tau_step = results_json["tau"], results_json["tau_error"]
    fbl_candidate, fbl_step = 1, 2 / search_space
    index = 1
    prev_min_chi2 = None
    output_dir = data_dir / "grid_search_4d"
    output_dir.mkdir(exist_ok=True)
    while True:
        click.echo(f"Grid search {index}")
        values_dict = dict(
            t0=create_search_list(t0_candidate, t0_step, search_space),
            u_min=create_search_list(u_min_candidate, u_min_step, search_space),
            tau=create_search_list(tau_candidate, tau_step, search_space),
            f_bl=create_search_list(fbl_candidate, fbl_step, search_space),
        )
        click.echo("Searching...")
        chi2_grid_table = grid_search(
            x=x,
            y=y,
            yerr=yerr,
            **values_dict,
        )
        click.echo("Found! saving results...")
        best_approximation = extract_grid_search_best_approximation(
            chi2_grid_table=chi2_grid_table
        )
        best_chi2 = best_approximation.pop("chi2")

        for parameter1, parameter2 in itertools.combinations(GRID_SEARCH_4D_NAMES, 2):
            chi2_grid = build_grid_matrix(
                chi2_grid_table, best_approximation, parameter1, parameter2
            )
            plot_grid(
                chi2_grid=chi2_grid,
                x_values=values_dict[parameter1],
                y_values=values_dict[parameter2],
                x_parameter=parameter1,
                y_parameter=parameter2,
                output_path=(
                    output_dir / f"grid_search_{parameter1}_{parameter2}{index}.png"
                ),
            )
        results = dict(best_approximation)
        results.update(
            calculate_errors_dict(
                chi2_grid_table, best_approximation, best_chi2=best_chi2
            )
        )
        results["chi2"] = best_chi2
        with open(output_dir / f"grid_search{index}_results.json", mode="w") as fd:
            json.dump(results, fd, indent=2)
        plot_fit(
            x=x,
            y_true=y,
            y_pred=calculate_intensity(t=x, **best_approximation),
            yerr=yerr,
            title=rf"Grid search fit ($\chi^2_{{red}} = {best_chi2:.2e}$)",
            xlabel="Time [sec]",
            ylabel="Intensity factor",
            output_file=output_dir / f"grid_search_fit{index}.png",
        )
        t0_candidate, u_min_candidate, tau_candidate, fbl_candidate = (
            results["t0"],
            results["u_min"],
            results["tau"],
            results["f_bl"],
        )
        click.echo(f"Min chi2: {best_chi2:.2e}")
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - best_chi2) / prev_min_chi2  # noqa: W503
            < chi2_epsilon  # noqa: W503
        ):
            break
        prev_min_chi2 = best_chi2
        t0_step, u_min_step, tau_step, fbl_step = (
            t0_step / 2,
            u_min_step / 2,
            tau_step / 2,
            fbl_step / 2,
        )
        index += 1
