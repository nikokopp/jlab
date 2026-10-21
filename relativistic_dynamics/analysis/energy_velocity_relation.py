import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.constants import c, electron_mass, electron_volt


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

    # Keep an optional field uncertainty if it exists in the energy CSV.
    optional_columns = [col for col in ("B_err_G",) if col in data.columns]
    data = data[COLUMNS + optional_columns].copy().sort_values("B_G").reset_index(drop=True)
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), layout="constrained")
    panels = (
        (axes[0], beta, beta_err, beta_curve,
         r"$\beta = v/c$", r"Kinetic energy vs. $\beta$"),
        (axes[1], gamma - 1, gamma_err, gamma_curve - 1,
         r"$\gamma - 1 = \frac{1}{\sqrt{1-\beta^2}} - 1$", r"Kinetic energy vs. $\gamma - 1$"),
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
        ax.set_xlim(left=0, right=1.0)
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



def calculate_mass_points(data):
    """Infer each electron's rest mass at each B, with pointwise uncertainties.

    Fits use K in keV; this function converts (mc^2) from keV to kilograms.
    Shared plate-gap, calibration, and detector-response systematics are
    not represented by these per-point uncertainties.
    """
    K = data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = data["K_combined_err_keV"].to_numpy(dtype=float)
    beta = data["beta"].to_numpy(dtype=float)
    beta_err = data["beta_err_measurement_approx"].to_numpy(dtype=float)
    gamma = data["gamma"].to_numpy(dtype=float)
    gamma_err = data["gamma_err_measurement_approx"].to_numpy(dtype=float)

    if np.any(K <= 0):
        raise ValueError("Positive kinetic energies are required to infer mass")

    # Each expression first yields the rest energy mc^2 in keV.
    mc2_rel_keV = K / (gamma - 1.0)
    mc2_nr_keV = 2.0 * K / beta**2

    # 1 keV of rest energy divided by c^2, in kg.
    kg_per_keV = 1000.0 * electron_volt / c**2
    m_rel = mc2_rel_keV * kg_per_keV
    m_nr = mc2_nr_keV * kg_per_keV

    # First-order propagation of independent pointwise K and beta/gamma errors.
    # Gamma errors are derived from beta errors and must NOT be included
    # as an additional independent term alongside beta errors.
    m_rel_err = m_rel * np.hypot(K_err / K, gamma_err / (gamma - 1.0))
    m_nr_err = m_nr * np.hypot(K_err / K, 2.0 * beta_err / beta)

    # Preserve the nominal field for tracking, but plot the corrected field.
    result = data[["B_G", "B_corrected_G", "B_err_G"]].copy()
    result["m_rel_kg"] = m_rel
    result["m_rel_err_pointwise_kg"] = m_rel_err
    result["m_nonrel_kg"] = m_nr
    result["m_nonrel_err_pointwise_kg"] = m_nr_err
    return result


def plot_mass_vs_field(mass_points, fits, output_png):
    """Analog of the e/m vs B figure: per-B estimates and fitted 1-sigma bands."""
    # Plot mass in units of 10^-31 kg to keep the tick labels readable.
    scale = 1e-31
    kg_per_keV = 1000.0 * electron_volt / c**2
    B = mass_points["B_corrected_G"].to_numpy(dtype=float)
    B_err = mass_points["B_err_G"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    styles = [
        ("nonrelativistic", "m_nonrel_kg", "m_nonrel_err_pointwise_kg",
         "steelblue", "#cfe3ff", "o", "Nonrelativistic"),
        ("relativistic", "m_rel_kg", "m_rel_err_pointwise_kg",
         "mediumpurple", "#e6d5ff", "s", "Relativistic"),
    ]

    # Render horizontal fitted-mass uncertainty bands first.
    for theory, col, err_col, point_color, band_color, marker, label in styles:
        fitted = fits[theory]["mc2_keV"] * kg_per_keV / scale
        sigma_fit = fits[theory]["mc2_err_stat_approx_keV"] * kg_per_keV / scale
        ax.axhspan(
            fitted - sigma_fit, fitted + sigma_fit,
            color=band_color, alpha=0.7, zorder=1,
            label=(rf"{label} fit: "
                   rf"$({fitted:.3f}\pm{sigma_fit:.3f}_\mathrm{{stat}})\times10^{{-31}}$ kg")
        )

    # Accepted electron rest mass.
    ax.axhline(
        electron_mass / scale, color="0.25", ls="--", lw=1.5,
        label=rf"Accepted $m_e$: ${electron_mass / scale:.3f}\times10^{{-31}}$ kg",
        zorder=2,
    )

    # Individual estimates at each B, with pointwise propagated y errors.
    for theory, col, err_col, point_color, band_color, marker, label in styles:
        ax.errorbar(
            B, mass_points[col].to_numpy(dtype=float) / scale,
            xerr=B_err,
            yerr=mass_points[err_col].to_numpy(dtype=float) / scale,
            fmt=marker, ms=5, capsize=3, elinewidth=1.1,
            color=point_color, ecolor=point_color,
            label=f"{label} points", zorder=4,
        )

    ax.set_xlabel(r"Magnetic field $B$ (G)")
    ax.set_ylabel(r"Electron mass $m_e$ ($10^{-31}$ kg)")
    ax.set_title("Calculated electron mass vs. magnetic field")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, framealpha=0.95, fontsize=9)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("data/processed/joint_energy_fit"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed/energy_velocity"))
    args = parser.parse_args()

    # Load the energy-summary measurements as before.
    data = load_data(args.results_dir, args.input_csv)

    # Load nominal field, corrected field, and field uncertainty from the
    # SAME processed trapezoid-fit file used by the momentum analysis.
    sweep_file = Path(
        "/home/rck67/jlab/relativistic_dynamics/data/processed/"
        "trapezoid_fits/velocity_sweep_results.csv"
    )
    sweep = pd.read_csv(
        sweep_file,
        usecols=["B_G", "B_corrected_G", "Berr_corrected_G", "V_opt_kV"],
    ).rename(columns={"Berr_corrected_G": "B_err_G"})

    # These new fields are authoritative for the x-axis. If the input energy
    # CSV happens to contain older copies, replace those instead of creating
    # duplicate _x / _y column names.
    data = data.drop(columns=["B_corrected_G", "B_err_G"], errors="ignore")
    data = data.merge(
        sweep,
        on="B_G",                  # Both files use nominal B_G as the key.
        how="left",
        validate="one_to_one",    # Catch duplicate nominal field settings.
        indicator=True,
    )
    unmatched = data.loc[data["_merge"] != "both", "B_G"]
    if not unmatched.empty:
        raise ValueError(
            "No velocity-sweep entry for nominal B_G values: "
            f"{unmatched.to_list()}"
        )
    data = data.drop(columns="_merge")
    if (data["B_err_G"] <= 0).any():
        raise ValueError("B_err_G must be positive")

    print("\nMatched magnetic-field measurements:")
    print(data[["B_G", "B_corrected_G", "B_err_G"]].to_string(index=False))

    # A useful consistency check: the existing energy-data beta must have
    # been calculated using the corrected field and optimized selector voltage.
    beta_from_sweep = (
        data["V_opt_kV"].to_numpy(dtype=float) * 1000.0
        / 299.792458 / 0.180
        / data["B_corrected_G"].to_numpy(dtype=float)
    )
    if not np.allclose(data["beta"], beta_from_sweep, rtol=0.01, atol=0):
        warnings.warn(
            "Energy-data beta differs by >1% from E/B calculated using "
            "the velocity sweep's optimized voltage and corrected B. "
            "Check upstream beta/gamma processing before interpreting mass."
        )

    # Save combined data with the corrected B columns.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = args.output_dir / "energy_velocity_results.csv"
    data.to_csv(combined_file, index=False)

    beta = data["beta"].to_numpy(dtype=float)
    beta_err = data["beta_err_measurement_approx"].to_numpy(dtype=float)
    K = data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = data["K_combined_err_keV"].to_numpy(dtype=float)

    print("\nExperimental measurements:")
    print(data[["B_G", "B_corrected_G", "K_deposited_keV",
                "K_combined_err_keV", "beta", "gamma"]].to_string(index=False))

    fits = {
        theory: fit_model(beta, beta_err, K, K_err, theory)
        for theory in ("relativistic", "nonrelativistic")
    }
    comparison = pd.DataFrame(list(fits.values()))
    comparison_file = args.output_dir / "energy_model_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\nFitted models:")
    print(comparison.to_string(index=False,
                               float_format=lambda value: f"{value:.3f}"))

    figure_file = Path(
        "/home/rck67/jlab/relativistic_dynamics/figures/"
        "relativistic_relations/energy_velocity_comparison.png"
    )
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    plot_results(data, fits, figure_file)

    mass_points = calculate_mass_points(data)
    mass_points_file = args.output_dir / "mass_by_field.csv"
    mass_points.to_csv(mass_points_file, index=False)

    mass_figure_file = figure_file.with_name("electron_mass_vs_B.png")
    plot_mass_vs_field(mass_points, fits, mass_figure_file)

    print("\nPointwise beta and K errors are included in the fits; common")
    print("plate-gap, correlated calibration, and detector-response")
    print("systematics are NOT included.")
    for path in (combined_file, comparison_file, figure_file,
                 mass_points_file, mass_figure_file):
        print(f"Saved: {path}")

# ======================================================================
# ======================================================================
#
#                       PAPER FIGURES
#             ENERGY-VELOCITY AND ELECTRON MASS
#
# ======================================================================
# ======================================================================

if __name__ == "__main__":
    # Use the processed results saved by main()
    # Respect --output-dir if one was specified
    paper_parser = argparse.ArgumentParser(add_help=False)
    paper_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/energy_velocity")
    )
    paper_args, _ = paper_parser.parse_known_args()

    results_dir = paper_args.output_dir

    paper_data = pd.read_csv(
        results_dir / "energy_velocity_results.csv"
    )

    fit_table = pd.read_csv(
        results_dir / "energy_model_comparison.csv"
    )
    paper_fits = fit_table.set_index("theory").to_dict("index")

    paper_mass = pd.read_csv(
        results_dir / "mass_by_field.csv"
    )

    paper_dir = Path(
        "/home/rck67/jlab/relativistic_dynamics/figures/paper_figures"
    )
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Common formatting
    label_size = 12
    tick_size = 10
    legend_size = 10
    stats_size = 9.5
    panel_size = 14


    # ==================================================================
    # FIGURE 1: ENERGY-VELOCITY RELATION
    # ==================================================================

    beta = paper_data["beta"].to_numpy(dtype=float)
    beta_err = paper_data[
        "beta_err_measurement_approx"
    ].to_numpy(dtype=float)

    gamma = paper_data["gamma"].to_numpy(dtype=float)
    gamma_err = paper_data[
        "gamma_err_measurement_approx"
    ].to_numpy(dtype=float)

    K = paper_data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = paper_data["K_combined_err_keV"].to_numpy(dtype=float)

    beta_max = max(0.9, min(0.97, float(beta.max()) + 0.07))
    beta_curve = np.linspace(0, beta_max, 600)
    gamma_curve = 1 / np.sqrt(1 - beta_curve**2)

    fig_energy, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(12.5, 4.5),
        sharey=True
    )

    fig_energy.subplots_adjust(
        left=0.085,
        right=0.992,
        top=0.90,
        bottom=0.155,
        wspace=0.08
    )

    panels = [
        (
            ax_a,
            beta,
            beta_err,
            beta_curve,
            r"$\beta = v/c$",
            "(a)"
        ),
        (
            ax_b,
            gamma - 1,
            gamma_err,
            gamma_curve - 1,
            r"$\gamma - 1 = \frac{1}{\sqrt{1-\beta^2}} - 1$",
            "(b)"
        )
    ]

    for ax, x, x_err, curve_x, xlabel, panel_label in panels:

        # Relativistic fit
        ax.plot(
            curve_x,
            K_model(
                beta_curve,
                paper_fits["relativistic"]["mc2_keV"],
                "relativistic"
            ),
            color="mediumpurple",
            lw=2,
            label="Relativistic fit"
        )

        # Nonrelativistic fit
        ax.plot(
            curve_x,
            K_model(
                beta_curve,
                paper_fits["nonrelativistic"]["mc2_keV"],
                "nonrelativistic"
            ),
            color="steelblue",
            lw=2,
            ls="--",
            label="Nonrelativistic fit"
        )

        # Measurements
        ax.errorbar(
            x, K,
            xerr=x_err,
            yerr=K_err,
            fmt="o",
            ms=4,
            color="midnightblue",
            ecolor="midnightblue",
            capsize=3,
            elinewidth=1.2,
            label="Measurements",
            zorder=5
        )

        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=tick_size)

        # Legend
        ax.legend(
            loc="upper left",
            fontsize=legend_size,
            framealpha=0.95
        )

        # Fit statistics
        ax.text(
            0.023, 0.73,
            fit_statistics_text(paper_fits),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=stats_size,
            linespacing=1.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
                "boxstyle": "round,pad=0.4"
            },
            zorder=8
        )

        # Panel labels outside top-left
        ax.text(
            0.0, 1.035,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=panel_size,
            fontweight="bold",
            clip_on=False
        )

    # Shared y-axis
    ax_a.set_ylabel(
        r"Kinetic energy $K$ (keV)",
        fontsize=label_size
    )
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="y", labelleft=False)

    # Save energy-velocity figure
    energy_pdf = paper_dir / "energy_velocity_paper.pdf"
    energy_png = paper_dir / "energy_velocity_paper.png"

    fig_energy.savefig(
        energy_pdf,
        bbox_inches="tight",
        pad_inches=0.02
    )
    fig_energy.savefig(
        energy_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Saved: {energy_pdf}")
    print(f"Saved: {energy_png}")


    # ==================================================================
    # FIGURE 2: ELECTRON MASS VS MAGNETIC FIELD
    # ==================================================================

    fig_mass, ax_mass = plt.subplots(figsize=(9.5, 5))

    fig_mass.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.965,
        bottom=0.27
    )

    mass_scale = 1e-31
    kg_per_keV = 1000.0 * electron_volt / c**2

    B = paper_mass["B_corrected_G"].to_numpy(dtype=float)
    B_err = paper_mass["B_err_G"].to_numpy(dtype=float)

    mass_styles = [
        (
            "nonrelativistic",
            "m_nonrel_kg",
            "m_nonrel_err_pointwise_kg",
            "steelblue",
            "#cfe3ff",
            "o",
            "Nonrelativistic"
        ),
        (
            "relativistic",
            "m_rel_kg",
            "m_rel_err_pointwise_kg",
            "mediumpurple",
            "#e6d5ff",
            "s",
            "Relativistic"
        )
    ]

    # ------------------------------------------------------------------
    # Fitted electron mass uncertainty bands
    # ------------------------------------------------------------------

    for theory, col, err_col, point_color, band_color, marker, label in mass_styles:

        fitted = (
            paper_fits[theory]["mc2_keV"]
            * kg_per_keV / mass_scale
        )

        sigma_fit = (
            paper_fits[theory]["mc2_err_stat_approx_keV"]
            * kg_per_keV / mass_scale
        )

        ax_mass.axhspan(
            fitted - sigma_fit,
            fitted + sigma_fit,
            color=band_color,
            alpha=0.65,
            zorder=0,
            label=(
                rf"{label} fit: "
                rf"$({fitted:.3f}\pm{sigma_fit:.3f}_\mathrm{{stat}})"
                r"\times10^{-31}$ kg"
            )
        )

    # ------------------------------------------------------------------
    # Accepted electron mass
    # ------------------------------------------------------------------

    ax_mass.axhline(
        electron_mass / mass_scale,
        color="0.25",
        linestyle="--",
        linewidth=1.5,
        label=(
            rf"Accepted $m_e$: "
            rf"${electron_mass / mass_scale:.3f}"
            r"\times10^{-31}$ kg"
        ),
        zorder=2
    )

    # ------------------------------------------------------------------
    # Individual electron mass measurements
    # ------------------------------------------------------------------

    for theory, col, err_col, point_color, band_color, marker, label in mass_styles:

        ax_mass.errorbar(
            B,
            paper_mass[col].to_numpy(dtype=float) / mass_scale,
            xerr=B_err,
            yerr=paper_mass[err_col].to_numpy(dtype=float) / mass_scale,
            fmt=marker,
            ms=5,
            color=point_color,
            ecolor=point_color,
            capsize=3,
            elinewidth=1.1,
            label=f"{label} points",
            zorder=4
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    ax_mass.set_xlabel(
        r"Magnetic field $B$ (G)",
        fontsize=label_size
    )

    ax_mass.set_ylabel(
        r"Electron mass $m_e$ ($10^{-31}$ kg)",
        fontsize=label_size
    )

    ax_mass.tick_params(labelsize=tick_size)
    ax_mass.grid(alpha=0.22)

    # Legend below the figure in two columns
    ax_mass.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=9.5,
        framealpha=0.95,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0
    )

    # ------------------------------------------------------------------
    # Save electron mass figure
    # ------------------------------------------------------------------

    mass_pdf = paper_dir / "electron_mass_vs_B_paper.pdf"
    mass_png = paper_dir / "electron_mass_vs_B_paper.png"

    fig_mass.savefig(
        mass_pdf,
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig_mass.savefig(
        mass_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Saved: {mass_pdf}")
    print(f"Saved: {mass_png}")


    # ==================================================================
    # PREVIEW PAPER FIGURES
    # ==================================================================

    plt.show()
