from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c, elementary_charge
from scipy.odr import Model, ODR, RealData

ROOT = Path("/home/rck67/jlab/relativistic_dynamics")
VELOCITY_CSV = ROOT / "data/processed/trapezoid_fits/velocity_sweep_results.csv"
ENERGY_DIR = ROOT / "data/processed/joint_energy_fit/"
COMBINED_ENERGY_CSV = ROOT / "data/processed/energy_velocity_results.csv"
OUTPUT_DIR = ROOT / "data/processed/energy_momentum"

# Orbit diameter and uncertainty from the lab manual (0.406 +/- 0.004 m).
rho_m = 0.406 / 2
rho_err_m = 0.004 / 2
MC2_ACCEPTED_KEV = 510.99895

colors = {
    "data": "midnightblue",
    "error": "midnightblue",
    "rel": "mediumpurple",
    "nonrel": "steelblue",
}

def load_energy_data():
    summary_files = sorted(ENERGY_DIR.rglob("energy_summary.csv"))
    if summary_files:
        frames = []
        for path in summary_files:
            frame = pd.read_csv(path)
            frame["source_file"] = str(path)
            frames.append(frame)
        energy = pd.concat(frames, ignore_index=True)
        print(f"Loaded {len(summary_files)} energy summaries from {ENERGY_DIR}")
    elif COMBINED_ENERGY_CSV.is_file():
        energy = pd.read_csv(COMBINED_ENERGY_CSV)
        print(f"Loaded previously combined energies from {COMBINED_ENERGY_CSV}")
    else:
        raise FileNotFoundError(
            "No energy_summary.csv files or combined energy_velocity_results.csv found.\n"
            f"Searched {ENERGY_DIR} and {COMBINED_ENERGY_CSV}"
        )

    needed = {"B_G", "K_deposited_keV", "K_combined_err_keV"}
    missing = needed - set(energy.columns)
    if missing:
        raise KeyError(f"Energy data missing columns: {sorted(missing)}; available: {list(energy.columns)}")
    return energy


def k_model(p_keV_c, mc2_keV, model):
    """p is in keV/c; p*c has the same numeric value in keV."""
    pc_keV = np.asarray(p_keV_c, dtype=float)
    if model == "rel":
        return np.hypot(pc_keV, mc2_keV) - mc2_keV
    if model == "nonrel":
        return pc_keV**2 / (2 * mc2_keV)
    raise ValueError(f"Unknown model: {model}")


def fit_mass(p, p_err_B, K, K_err, model):
    """One-parameter orthogonal distance regression in independent (p,K).

    Only field-related momentum errors enter the fit. The orbit-radius
    uncertainty is common to all points, so it is not an independent x error.
    Other correlated calibration errors are not modeled in this fit.
    """
    def model_function(log_mass, x):
        return k_model(x, np.exp(log_mass[0]), model)

    real_data = RealData(p, K, sx=p_err_B, sy=K_err)
    output = ODR(real_data, Model(model_function), beta0=[np.log(MC2_ACCEPTED_KEV)]).run()
    if output.info > 4:
        raise RuntimeError(f"{model} fit failed: {output.stopreason}")

    mc2 = float(np.exp(output.beta[0]))
    mc2_err = float(mc2 * np.sqrt(output.cov_beta[0, 0]))
    chi2 = float(output.sum_square)
    dof = len(p) - 1
    return {"model": model, "mc2_keV": mc2, "mc2_err_keV": mc2_err,
            "chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof}


def main():
    velocity = pd.read_csv(VELOCITY_CSV)
    energy = load_energy_data()

    required_velocity = {"B_G", "B_corrected_G", "Berr_corrected_G"}
    missing = required_velocity - set(velocity.columns)
    if missing:
        raise KeyError(f"Velocity CSV missing columns: {sorted(missing)}")

    # B_G is an identifier (nominal/file field). IMPORTANT: use the corrected
    # measured field, not B_G, in the physics calculation below.
    velocity = velocity[["B_G", "B_corrected_G", "Berr_corrected_G"]].copy()
    energy = energy.copy()
    velocity["B_G"] = pd.to_numeric(velocity["B_G"], errors="raise").round(3)
    energy["B_G"] = pd.to_numeric(energy["B_G"], errors="raise").round(3)

    if energy["B_G"].duplicated().any() or velocity["B_G"].duplicated().any():
        raise ValueError(
            "Duplicate B_G run labels detected. Check that only the intended seven "
            "energy summaries are included and that each sweep appears once."
        )

    energy = energy.drop(
        columns=["B_corrected_G", "Berr_corrected_G"],
        errors="ignore"
    )

    data = energy.merge(velocity, on="B_G", how="left", validate="one_to_one", indicator=True)
    unmatched = data.loc[data["_merge"] != "both", "B_G"].tolist()
    if unmatched:
        raise ValueError(f"Unmatched B_G identifiers: {unmatched}. Check the sweep and energy run labels.")
    data = data.drop(columns="_merge").sort_values("B_corrected_G")

    B = data["B_corrected_G"].to_numpy(float)    # gauss, corrected
    B_err = data["Berr_corrected_G"].to_numpy(float)
    K = data["K_deposited_keV"].to_numpy(float)
    K_err = data["K_combined_err_keV"].to_numpy(float)
    if len(B) < 3 or np.any(~np.isfinite(np.r_[B, B_err, K, K_err])):
        raise ValueError("Need >=3 matched finite observations with valid errors.")
    if np.any(B <= 0) or np.any(B_err <= 0) or np.any(K_err <= 0):
        raise ValueError("Magnetic fields and uncertainties must be positive.")

    # SI magnetic rigidity p = |e| B rho. Convert p to keV/c for the x-axis.
    p_SI = elementary_charge * (B * 1e-4) * rho_m  # kg m/s
    p_keV_c = p_SI * c / (elementary_charge * 1e3)

    # Field-related uncertainty is included in orthogonal-distance fits.
    p_err_B = p_keV_c * (B_err / B)
    # Radius error is shared by all points, so show it but DO NOT treat it as
    # independent for each point during fitting.
    p_err_rho = p_keV_c * (rho_err_m / rho_m)
    p_err_total = np.hypot(p_err_B, p_err_rho)

    fit_rel = fit_mass(p_keV_c, p_err_B, K, K_err, "rel")
    fit_nr = fit_mass(p_keV_c, p_err_B, K, K_err, "nonrel")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data["p_keV_c"] = p_keV_c
    data["p_err_B_keV_c"] = p_err_B
    data["p_err_rho_common_keV_c"] = p_err_rho
    data["p_err_total_keV_c"] = p_err_total
    data["K_rel_fit_keV"] = k_model(p_keV_c, fit_rel["mc2_keV"], "rel")
    data["K_nonrel_fit_keV"] = k_model(p_keV_c, fit_nr["mc2_keV"], "nonrel")
    data.to_csv(OUTPUT_DIR / "energy_momentum_results.csv", index=False)
    pd.DataFrame([fit_rel, fit_nr]).to_csv(OUTPUT_DIR / "energy_momentum_model_comparison.csv", index=False)

    p_grid = np.linspace(0, max(p_keV_c + p_err_total) * 1.07, 600)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.errorbar(p_keV_c, K, xerr=p_err_total, yerr=K_err,
                fmt="o", ms=4, capsize=3, color=colors["data"],
                ecolor=colors["error"], elinewidth=1.1, label="Measurements")
    ax.plot(p_grid, k_model(p_grid, fit_rel["mc2_keV"], "rel"),
            color=colors["rel"], lw=2, label="Relativistic fit")
    ax.plot(p_grid, k_model(p_grid, fit_nr["mc2_keV"], "nonrel"),
            color=colors["nonrel"], lw=2, ls="--", label="Nonrelativistic fit")

    ax.set_title("Kinetic energy vs. momentum")
    ax.set_xlabel(r"Momentum $p$ (keV/$c$)")
    ax.set_ylabel(r"Kinetic energy $K$ (keV)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=9)
    stats = (
        rf"Relativistic fit: $\chi^2/\mathrm{{dof}}={fit_rel['chi2']:.3f}/{fit_rel['dof']}="
        rf"{fit_rel['chi2_dof']:.3f}$" + "\n" +
        rf"Nonrelativistic fit: $\chi^2/\mathrm{{dof}}={fit_nr['chi2']:.3f}/{fit_nr['dof']}="
        rf"{fit_nr['chi2_dof']:.3f}$"
    )
    ax.text(0.98, 0.055, stats, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=9, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                            "edgecolor": "0.7", "alpha": 0.9})
    fig.tight_layout()
    figure_file = ROOT / "figures/relativistic_relations/energy_momentum_comparison.png"
    fig.savefig(figure_file, dpi=300, bbox_inches="tight")

    print("\nMomentum and deposited kinetic energy:")
    print(data[["B_G", "B_corrected_G", "p_keV_c", "p_err_total_keV_c",
                "K_deposited_keV", "K_combined_err_keV"]].to_string(
                    index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nFitted mc^2 (one parameter per model):")
    for result in (fit_rel, fit_nr):
        print(f" {result['model']:6s}: mc^2 = {result['mc2_keV']:.2f} +/- "
              f"{result['mc2_err_keV']:.2f} keV; "
              f"chi2/dof = {result['chi2']:.2f}/{result['dof']} = "
              f"{result['chi2_dof']:.2f}")
    print(f" Accepted electron mc^2 = {MC2_ACCEPTED_KEV:.5f} keV")
    print("NOTE: the radius uncertainty is shared across points and omitted from fit weights;")
    print("      chi2 does not include all possible correlated calibration/systematic errors.")
    print(f"\nSaved plot: {figure_file}\nSaved data and model comparison: {OUTPUT_DIR}")
    plt.show()

# ======================================================================
# ======================================================================
#
#                       PAPER FIGURE
#                  ENERGY-MOMENTUM RELATION
#
# ======================================================================
# ======================================================================

if __name__ == "__main__":

    paper_dir = ROOT / "figures/paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Load previously calculated data and fit results
    paper_data = pd.read_csv(
        OUTPUT_DIR / "energy_momentum_results.csv"
    )

    fit_table = pd.read_csv(
        OUTPUT_DIR / "energy_momentum_model_comparison.csv"
    )
    paper_fits = fit_table.set_index("model").to_dict("index")

    fit_rel = paper_fits["rel"]
    fit_nr = paper_fits["nonrel"]

    # Extract measurements
    p = paper_data["p_keV_c"].to_numpy(dtype=float)
    p_err = paper_data["p_err_total_keV_c"].to_numpy(dtype=float)
    K = paper_data["K_deposited_keV"].to_numpy(dtype=float)
    K_err = paper_data["K_combined_err_keV"].to_numpy(dtype=float)

    # Momentum range for fitted curves
    p_grid = np.linspace(
        0,
        max(p + p_err) * 1.07,
        600
    )

    # ==================================================================
    # FIGURE: KINETIC ENERGY VS MOMENTUM
    # ==================================================================

    fig_paper, ax = plt.subplots(figsize=(8.5, 4.5))

    # Minimize surrounding whitespace
    fig_paper.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.975,
        bottom=0.135
    )

    label_size = 12
    tick_size = 10
    legend_size = 10
    stats_size = 9.5

    # ------------------------------------------------------------------
    # Experimental measurements
    # ------------------------------------------------------------------

    ax.errorbar(
        p, K,
        xerr=p_err,
        yerr=K_err,
        fmt="o",
        ms=4,
        color=colors["data"],
        ecolor=colors["error"],
        capsize=3,
        elinewidth=1.2,
        label="Measurements",
        zorder=5
    )

    # ------------------------------------------------------------------
    # Relativistic fit
    # ------------------------------------------------------------------

    ax.plot(
        p_grid,
        k_model(p_grid, fit_rel["mc2_keV"], "rel"),
        color=colors["rel"],
        lw=2,
        label="Relativistic fit"
    )

    # ------------------------------------------------------------------
    # Nonrelativistic fit
    # ------------------------------------------------------------------

    ax.plot(
        p_grid,
        k_model(p_grid, fit_nr["mc2_keV"], "nonrel"),
        color=colors["nonrel"],
        lw=2,
        ls="--",
        label="Nonrelativistic fit"
    )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    ax.set_xlabel(
        r"Momentum $p$ (keV/$c$)",
        fontsize=label_size
    )

    ax.set_ylabel(
        r"Kinetic energy $K$ (keV)",
        fontsize=label_size
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.22)

    # ------------------------------------------------------------------
    # Legend: original upper-left position
    # ------------------------------------------------------------------

    ax.legend(
        loc="upper left",
        fontsize=legend_size,
        framealpha=0.95
    )

    # ------------------------------------------------------------------
    # Fit statistics: positioned below legend
    # ------------------------------------------------------------------

    stats_text = (
        rf'Relativistic fit: $\chi^2/\mathrm{{dof}} = '
        rf'{fit_rel["chi2"]:.3f}/{fit_rel["dof"]} = '
        rf'{fit_rel["chi2_dof"]:.3f}$'
        "\n"
        rf'Nonrelativistic fit: $\chi^2/\mathrm{{dof}} = '
        rf'{fit_nr["chi2"]:.3f}/{fit_nr["dof"]} = '
        rf'{fit_nr["chi2_dof"]:.3f}$'
    )

    ax.text(
        0.016, 0.76,
        stats_text,
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

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    pdf_file = paper_dir / "energy_momentum_paper.pdf"
    png_file = paper_dir / "energy_momentum_paper.png"

    fig_paper.savefig(
        pdf_file,
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig_paper.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Saved PDF: {pdf_file}")
    print(f"Saved PNG: {png_file}")

    plt.show()