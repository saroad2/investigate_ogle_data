import json
from pathlib import Path

import click
import numpy as np
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.constants import CHI2_EPSILON, DATA_FILE_NAME, DEFAULT_SPACE_SEARCH
from ogle.grid_search import extract_grid_search_2d_results, grid_search_2d
from ogle.io_util import read_data
from ogle.plot_util import plot_2d_grid


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
def grid_search_2d_cli(data_dir, tau, fbl, search_space):
    data_path = data_dir / f"{DATA_FILE_NAME}.csv"
    _, x, y = read_data(data_path=data_path, is_random=False)
    with (data_dir / f"{DATA_FILE_NAME}_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    t0_candidate, t0_step = results_json["t0"][:2]
    u_min_candidate, u_min_step = results_json["u_min"][:2]
    index = 1
    prev_min_chi2 = None
    while True:
        click.echo(f"Grid search {index}")
        t0_values = [
            t0_candidate + n * t0_step
            for n in range(-search_space // 2, search_space // 2)
        ]
        u_min_values = [
            u_min_candidate + n * u_min_step
            for n in range(-search_space // 2, search_space // 2)
        ]
        chi2_grid = grid_search_2d(
            x, y, t0_values=t0_values, u_min_values=u_min_values, tau=tau, f_bl=fbl
        )
        output_dir = data_dir / "grid_search_2d"
        output_dir.mkdir(exist_ok=True)
        plot_2d_grid(
            chi2_grid=chi2_grid,
            t0_values=t0_values,
            u_min_values=u_min_values,
            output_path=output_dir / f"grid_search{index}.png",
        )
        results = extract_grid_search_2d_results(
            chi2_grid=chi2_grid, t0_values=t0_values, u_min_values=u_min_values
        )
        with open(output_dir / f"grid_search{index}_results.json", mode="w") as fd:
            json.dump(results, fd, indent=2)
        min_chi2 = results["chi2"]
        t0_candidate, u_min_candidate = results["t0"], results["u_min"]
        click.echo(f"Min chi2: {min_chi2:.2e}")
        if (
            prev_min_chi2 is not None
            and np.fabs(prev_min_chi2 - min_chi2) < CHI2_EPSILON  # noqa: W503
        ):
            break
        prev_min_chi2 = min_chi2
        t0_step, u_min_step = t0_step / 2, u_min_step / 2
        index += 1
