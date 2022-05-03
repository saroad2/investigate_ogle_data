import json

import numpy as np
from matplotlib import pyplot as plt
from ogle.ogle_util import extract_microlensing_properties

MICROLENSING_PROPERTY_NAMES = ["t0", "f_max", "u_min"]


def plot_parabolic_fit(x, y, fit_result, t_start, output_dir, real_a=None):
    output_dir.mkdir(exist_ok=True)
    microlensing_properties = extract_microlensing_properties(
        a=fit_result.a, aerr=fit_result.aerr, t_start=t_start
    )
    plt.title(rf"Parabolic fit ($\chi^2_{{red}} = {fit_result.chi2_reduced:.2e}$)")
    plt.scatter(x, y, label="Data points")
    plt.plot(x, np.polyval(fit_result.a, x - t_start), label="Evaluated parabola")
    if real_a is not None:
        plt.plot(x, np.polyval(real_a, x - t_start), label="Real parabola")
    plt.legend()
    plt.savefig(output_dir / "parabolic_fit.png")
    plt.clf()

    with open(output_dir / "fit_result.json", mode="w") as fd:
        result_as_dict = fit_result.as_dict()
        result_as_dict.update(
            {
                key: [
                    value.nominal_value,
                    value.std_dev,
                    value.std_dev / np.fabs(value.nominal_value) * 100,
                ]
                for key, value in microlensing_properties.items()
            }
        )
        json.dump(result_as_dict, fd, indent=2)


def plot_monte_carlo_results(results, output_dir):
    output_dir.mkdir(exist_ok=True)
    for property_name in MICROLENSING_PROPERTY_NAMES:
        a = np.array([result[property_name].nominal_value for result in results])
        a.sort()
        aerr = np.array([result[property_name].std_dev for result in results])
        mean_value = np.mean(a)
        sample_error = np.sqrt(np.sum(aerr**2)) / a.shape[0]
        stat_error = np.std(a)
        total_error = np.sqrt(sample_error**2 + stat_error**2)
        percentage_error = total_error / np.fabs(mean_value) * 100
        plt.title(
            f"Values hist for {property_name} - {mean_value:.2e} "
            rf"$\pm$ {total_error:.2e} "
            f"( {percentage_error:.2f}% )"
        )
        plt.xlabel(f"{property_name} values")
        plt.ylabel("Count")
        max_hist = np.max(np.histogram(a, bins=50)[0])
        plt.hist(a, bins=50)
        plt.plot(a, max_hist * np.exp(-(((a - mean_value) / total_error) ** 2)))
        plt.savefig(output_dir / f"{property_name}_hist.png")
        plt.clf()
    for i in range(len(MICROLENSING_PROPERTY_NAMES) - 1):
        for j in range(i + 1, len(MICROLENSING_PROPERTY_NAMES)):
            property_name1, property_name2 = (
                MICROLENSING_PROPERTY_NAMES[i],
                MICROLENSING_PROPERTY_NAMES[j],
            )
            x = np.array([result[property_name1].nominal_value for result in results])
            y = np.array([result[property_name2].nominal_value for result in results])
            covariance = np.cov(x, y)[0, 1]
            correlation = covariance / (np.mean(x) * np.mean(y))
            plt.title(
                f"Correlation for {property_name1} and {property_name2} "
                f"- {correlation:.2f}"
            )
            plt.scatter(x, y)
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
    c = plt.contour(X, Y, chi2_grid, colors="yellow", linestyles="dashed")
    c.clabel(inline=True)
    i, j = np.unravel_index(chi2_grid.argmin(), chi2_grid.shape)
    plt.plot(
        [t0_values[i]], [u_min_values[j]], linestyle="none", marker="o", color="yellow"
    )
    plt.xlabel("$t_0$")
    plt.ylabel("$u_{min}$")
    plt.savefig(output_path)
    plt.clf()
