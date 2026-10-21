import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


COLUMNS = [
    "B_G", "date", "K_deposited_keV", "K_stat_err_keV",
    "K_cal_err_keV", "K_combined_err_keV", "beta",
    "beta_err_measurement_approx", "beta_err_gap_common_sys",
    "gamma", "gamma_err_measurement_approx", "gamma_err_gap_common_sys",
]


def load_data(results_dir, input_csv):
    if input_csv is not None:
        data = pd.read_csv(input_csv)
        print(f"Reading {input_csv}")
    else:
        files = sorted(results_dir.glob("B*_*/energy_summary.csv"))
        if not files:
            raise FileNotFoundError(f"No B*_*/energy_summary.csv in {results_dir}")
        print(f"Reading {len(files)} summary files:")
        for path in files:
            print(f"  {path}")
        data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    missing = [col for col in COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[COLUMNS].copy().sort_values("B_G").reset_index(drop=True)
    if data["B_G"].duplicated().any():
        raise ValueError("Duplicate B_G values: check which field/day results were loaded")
    if len(data) != 7:
        warnings.warn(f"Expected seven field values, found {len(data)}")

    numeric_columns = [col for col in COLUMNS if col != "date"]
    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Input data has a missing or nonfinite value")
    for col in ("K_combined_err_keV", "beta_err_measurement_approx"):
        if (data[col] <= 0).any():
            raise ValueError(f"{col} must be positive")
    if not data["beta"].between(0, 1, inclusive="neither").all():
        raise ValueError("Expected 0 < beta < 1")

    gamma_check = 1 / np.sqrt(1 - data["beta"].to_numpy() ** 2)
    if not np.allclose(gamma_check, data["gamma"], rtol=2e-4):
        raise ValueError("Input beta and gamma are inconsistent")
    return data


def K_model(beta, mc2, theory):
    """Predicted incident kinetic energy, in keV."""
    if theory == "relativistic":
        return mc2 * (1 / np.sqrt(1 - beta**2) - 1)
    if theory == "nonrelativistic":
        return 0.5 * mc2 * beta**2
    raise ValueError(f"Unknown theory: {theory}")


def fit_model(beta, beta_err, K, K_err, theory):
    """Errors-in-variables fit to beta and energy simultaneously.

    Parameters are mc2 and N latent true beta values. There are 2N measured
    quantities (N energies and N velocities), so dof = 2N - (N+1) = N-1.
    This is a Gaussian-error profile fit and treats pointwise errors as
    independent. Common systematic uncertainties are not incorporated.
    """
    n = len(beta)
    if n < 2:
        raise ValueError("Need at least two points to fit mc2")

    # Estimate a starting slope from the data, not a fixed accepted-mass curve.
    shape = K_model(beta, 1.0, theory)
    mc2_initial = max(1.0, float(np.sum(shape * K) / np.sum(shape**2)))
    start = np.r_[mc2_initial, beta]

    def residuals(params):
        mc2, beta_true = params[0], params[1:]
        r_K = (K - K_model(beta_true, mc2, theory)) / K_err
        r_beta = (beta - beta_true) / beta_err
        return np.r_[r_K, r_beta]

    result = least_squares(
        residuals,
        start,
        bounds=(np.r_[1e-8, np.zeros(n)],
                np.r_[np.inf, np.full(n, 0.999999)]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=20000,
    )
    if not result.success:
        warnings.warn(f"{theory} fit did not converge: {result.message}")

    chi2 = float(np.dot(result.fun, result.fun))
    dof = n - 1

    # Approximate 1-sigma uncertainty from local fit curvature, conditional
    # on pointwise uncertainties and this particular response model.
    covariance = np.linalg.pinv(result.jac.T @ result.jac)
    mc2_err = float(np.sqrt(max(covariance[0, 0], 0)))
    return {
        "theory": theory,
        "mc2_keV": float(result.x[0]),
        "mc2_err_stat_approx_keV": mc2_err,
        "chi2": chi2,
        "dof": dof,
        "chi2_over_dof": chi2 / dof,
    }


def fit_statistics_text(fits):
    """Compact annotation repeated in both coordinate representations."""
    rel = fits["relativistic"]
    nr = fits["nonrelativistic"]
    return (
        "Relativistic fit:"
        f"  χ²/dof = {rel['chi2']:.3f}/{rel['dof']} = {rel['chi2_over_dof']:.3f}"
        "\n"
        "Nonrelativistic fit:"
        f"  χ²/dof = {nr['chi2']:.3f}/{nr['dof']} = {nr['chi2_over_dof']:.3f}"
    )


def plot_results(data, fits, output_png):
    beta = data["beta"].to_numpy(dtype=float)
    beta_err = data["beta_err_measurement_approx"].to_numpy(dtype=float)
    gamma = data["gamma"].to_numpy(dtype=float)
    gamma_err = data["gamma_err_measurement_approx"].to_numpy(dtype=float)
    K = data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = data["K_combined_err_keV"].to_numpy(dtype=float)

    beta_max = max(0.9, min(0.97, float(beta.max()) + 0.07))
    beta_curve = np.linspace(0, beta_max, 600)
    gamma_curve = 1 / np.sqrt(1 - beta_curve**2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), layout="constrained")
    panels = (
        (axes[0], beta, beta_err, beta_curve,
         r"$\beta = v/c$", r"Kinetic energy vs. $\beta$"),
        (axes[1], gamma - 1, gamma_err, gamma_curve - 1,
         r"$\gamma - 1 = \sqrt{1-\beta^2} - 1$", r"Kinetic energy vs. $\gamma - 1$"),
    )

    for ax, x, x_err, curve_x, xlabel, title in panels:
        for theory, color, linestyle, label in (
            ("relativistic", "mediumpurple", "-", "Relativistic fit"),
            ("nonrelativistic", "steelblue", "--", "Nonrelativistic fit"),
        ):
            fitted_mass = fits[theory]["mc2_keV"]
            ax.plot(
                curve_x, K_model(beta_curve, fitted_mass, theory),
                color=color, lw=1.8, ls=linestyle,
                label=label,
            )

        ax.errorbar(
            x, K, xerr=x_err, yerr=K_err,
            fmt="o", markersize=4, capsize=3,
            color="midnightblue", label="Measurements", zorder=5,
        )
        ax.set(title=title, xlabel=xlabel, ylabel=r"Kinetic energy $K$ (keV)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)

        # Legend first; the concise stats box sits immediately beneath it.
        ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985),
                  fontsize=9, framealpha=0.95)
        ax.text(
            0.98, 0.055, fit_statistics_text(fits),
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.7", alpha=0.9),
            zorder=8,
        )

    fig.savefig(output_png, dpi=220)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("data/processed/joint_energy_fit"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/energy_velocity"))
    args = parser.parse_args()

    data = load_data(args.results_dir, args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = args.output_dir / "energy_velocity_results.csv"
    data.to_csv(combined_file, index=False)

    beta = data["beta"].to_numpy(dtype=float)
    beta_err = data["beta_err_measurement_approx"].to_numpy(dtype=float)
    K = data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = data["K_combined_err_keV"].to_numpy(dtype=float)

    print("\nExperimental measurements:")
    print(data[["B_G", "K_deposited_keV", "K_combined_err_keV",
                "beta", "gamma"]].to_string(index=False))

    fits = {
        theory: fit_model(beta, beta_err, K, K_err, theory)
        for theory in ("relativistic", "nonrelativistic")
    }
    comparison = pd.DataFrame(list(fits.values()))
    comparison_file = args.output_dir / "energy_model_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\nFitted models (the same fits shown in both coordinate systems):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    figure_file = "/home/rck67/jlab/relativistic_dynamics/figures/relativistic_relations/energy_velocity_comparison.png"
    plot_results(data, fits, figure_file)

    print("\nχ², dof and χ²/dof are shown on BOTH plots. These are the same")
    print("measurements transformed from β to γ−1, not independent fits.")
    print("Pointwise β and K errors are included; common plate-gap uncertainty,")
    print("correlated calibration, spectral-model and detector-response")
    print("systematics are NOT included.")
    print(f"\nSaved: {combined_file}")
    print(f"Saved: {comparison_file}")
    print(f"Saved: {figure_file}")


if __name__ == "__main__":
    main()
