import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from ogle.ogle_util import extract_microlensing_properties
from scipy.integrate import trapz
from scipy.stats import norm


def plot_parabolic_fit(x, y, yerr, fit_result, t_start, output_dir):
    output_dir.mkdir(exist_ok=True)
    microlensing_properties = extract_microlensing_properties(
        a=fit_result.a, aerr=fit_result.aerr, t_start=t_start
    )
    plot_fit(
        x=x,
        y_true=y,
        y_pred=np.polyval(fit_result.a, x - t_start),
        yerr=yerr,
        title=rf"Parabolic fit ($\chi^2_{{red}} = {fit_result.chi2_reduced:.2e}$)",
        xlabel="Time [sec]",
        ylabel="Intensity factor",
        output_file=output_dir / "parabolic_fit.png",
    )

    with open(output_dir / "fit_result.json", mode="w") as fd:
        result_as_dict = fit_result.as_dict()
        result_as_dict.update(microlensing_properties)
        json.dump(result_as_dict, fd, indent=2)


def plot_fit(
    x,
    y_true,
    y_pred,
    yerr,
    title,
    xlabel,
    ylabel,
    output_file,
):
    plt.title(title)
    plt.errorbar(x, y_true, yerr=yerr, label="Data points", linestyle="none")
    plt.plot(x, y_pred, label="Evaluated fit")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_file)
    plt.clf()


def plot_monte_carlo_results(
    results, property_names, output_dir, normal_curve: bool = True, bins: int = 50
):
    output_dir.mkdir(exist_ok=True)
    results_dict = {}
    for property_name in property_names:
        results_dict[property_name] = [result[property_name] for result in results]
        results_dict[f"{property_name}_error"] = [
            result[f"{property_name}_error"] for result in results
        ]
    with open(output_dir / "results.json", mode="w") as fd:
        json.dump(results_dict, fd, indent=2)
    for property_name in property_names:
        a = np.array(results_dict[property_name])
        a.sort()
        mu, sigma = norm.fit(a)
        percentage_error = sigma / np.fabs(mu) * 100
        plt.title(
            f"Values histogram for {property_name} - {mu:.2e} "
            rf"$\pm$ {sigma:.2e} "
            f"( {percentage_error:.2f}% )"
        )
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        plt.xlabel(f"{property_name} values")
        plt.ylabel("Count")
        hist_heights, bins_edges, _ = plt.hist(
            a, bins=bins, label=f"Histogram of {a.shape[0]} samples"
        )
        if normal_curve:
            density = trapz(hist_heights, bins_edges[1:])
            plt.plot(
                bins_edges,
                norm.pdf(bins_edges, mu, sigma) * density,
                label="Normal distribution fit",
            )
        plt.legend()
        plt.savefig(output_dir / f"{property_name}_hist.png")
        plt.clf()
    for i in range(len(property_names) - 1):
        for j in range(i + 1, len(property_names)):
            property_name1, property_name2 = (
                property_names[i],
                property_names[j],
            )
            x = np.array([result[property_name1] for result in results])
            y = np.array([result[property_name2] for result in results])
            covariance = np.cov(x, y)[0, 1]
            plt.title(
                f"Covariance for {property_name1} and {property_name2} "
                f"- {covariance:.2e}"
            )
            plt.scatter(x, y)
            plt.xlabel(property_name1)
            plt.ylabel(property_name2)
            plt.savefig(
                output_dir / f"{property_name1}_{property_name2}_correlation.png"
            )
            plt.clf()


def plot_2d_grid(chi2_grid, t0_values, u_min_values, output_path):
    x_min, x_max, y_min, y_max = (
        t0_values[0],
        t0_values[-1],
        u_min_values[0],
        u_min_values[-1],
    )
    heatmap = plt.imshow(
        chi2_grid, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto"
    )
    plt.colorbar(heatmap)
    X, Y = np.meshgrid(t0_values, u_min_values)
    i, j = np.unravel_index(chi2_grid.argmin(), chi2_grid.shape)
    best_t0, best_u_min = t0_values[i], u_min_values[j]
    best_chi2 = chi2_grid[i, j]
    plt.contour(
        X,
        Y,
        chi2_grid,
        colors=["green", "yellow", "red"],
        linestyles="dashed",
        levels=[best_chi2 + 2.3, best_chi2 + 4.61, best_chi2 + 9.21],
    )
    plt.plot(
        [best_t0],
        [best_u_min],
        linestyle="none",
        marker="o",
        color="yellow",
        label="Best approximation",
    )
    plt.xlabel("$t_0$")
    plt.ylabel("$u_{min}$")
    plt.title(r"Grid search $\chi^2$ map")
    plt.legend(
        handles=[
            Line2D(
                [0], [0], color="green", lw=1, ls="dashed", label=r"$\Delta\chi^2=2.3$"
            ),
            Line2D(
                [0],
                [0],
                color="yellow",
                lw=1,
                ls="dashed",
                label=r"$\Delta\chi^2=4.61$",
            ),
            Line2D(
                [0], [0], color="red", lw=1, ls="dashed", label=r"$\Delta\chi^2=9.21$"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                ls="none",
                markerfacecolor="yellow",
                label=rf"Best $\chi2={best_chi2:.2f}$",
            ),
        ]
    )
    plt.savefig(output_path)
    plt.clf()
