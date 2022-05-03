import json
from pathlib import Path

import click
from matplotlib import pyplot as plt
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.constants import DATA_FILE_NAME
from ogle.io_util import read_data
from ogle.ogle_util import calculate_intensity


@ogle_cli_group.group("grid-search")
def grid_search_cli_group():
    """Grid-search related commands."""


@grid_search_cli_group.command("2d-search")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--tau", type=float, required=True)
@click.option("--fbl", type=float, default=1)
def grid_search_2d(data_dir, tau, fbl):
    data_path = data_dir / f"{DATA_FILE_NAME}.csv"
    _, x, y = read_data(data_path=data_path, is_random=False)
    with (data_dir / f"{DATA_FILE_NAME}_fitting_results" / "fit_Result.json").open(
        mode="r"
    ) as fd:
        results_json = json.load(fd)
    start_t0, t0_step = results_json["t0"][:2]
    start_u_min, u_min_step = results_json["u_min"][:2]
    intensity = calculate_intensity(
        t=x, t0=start_t0, tau=tau, u_min=start_u_min, f_bl=fbl
    )
    plt.plot(x, intensity)
    # plt.scatter(x, y, s=0.1)
    plt.show()
