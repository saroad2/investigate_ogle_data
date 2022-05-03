import json
from pathlib import Path

import click
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.constants import DATA_FILE_NAME, DEFAULT_SPACE_SEARCH
from ogle.grid_search import grid_search_2d
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
    start_t0, t0_step = results_json["t0"][:2]
    start_u_min, u_min_step = results_json["u_min"][:2]
    t0_values = [
        start_t0 + n * t0_step for n in range(-search_space // 2, search_space // 2)
    ]
    u_min_values = [
        start_u_min + n * u_min_step
        for n in range(-search_space // 2, search_space // 2)
    ]
    chi2_grid = grid_search_2d(
        x, y, t0_values=t0_values, u_min_values=u_min_values, tau=tau, f_bl=fbl
    )
    plot_2d_grid(chi2_grid=chi2_grid, t0_values=t0_values, u_min_values=u_min_values)
