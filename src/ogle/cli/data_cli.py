from pathlib import Path

import click
import pandas as pd
from matplotlib import pyplot as plt
from ogle.cli.ogle_cli import ogle_cli_group
from ogle.io_util import build_data, search_data_paths


@ogle_cli_group.command("build-data")
@click.argument("data_path", type=click.Path(path_type=Path, exists=True))
@click.option("--show/--no-show", is_flag=True, default=False)
def build_data_cli(data_path, show):
    data_paths = search_data_paths(data_path, suffix="dat")
    for i, path in enumerate(data_paths, start=1):
        click.echo(f"Build data for {path} ({i}/{len(data_paths)})")
        x, y, y_err = build_data(path)
        if show:
            plt.errorbar(x, y, yerr=y_err, linestyle="none")
            plt.show()
            plt.clf()
        output_path = path.with_name(f"{path.stem}.csv")
        pd.DataFrame(dict(x=x, y=y, y_err=y_err)).to_csv(
            output_path, index=False, header=True
        )
