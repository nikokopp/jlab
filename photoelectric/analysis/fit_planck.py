import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.constants import c, e

output_dir = Path("/home/rck67/jlab/photoelectric/")
figure_dir = output_dir / "figures/planck_fits"
figure_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / "data/planck_fit_results.csv"

method_files = {
    "Method 1": output_dir / "data/method1_dubridge_cutoff_summary.csv",
    "Method 2": output_dir / "data/method2_piecewise_cutoff_summary.csv",
}

palette = ["mediumorchid", "cornflowerblue", "indigo", "navy"]
colors = dict(zip(method_files.keys(), palette[0:2]))
data_colors = dict(zip(method_files.keys(), palette[2:]))
def linear(x, m, b):
    return m*x + b

def fit_planck(summary_file, method_name):
    """Fit mean cutoff voltage versus frequency for one cutoff extraction method."""
    cutoffs = pd.read_csv(summary_file).copy()
    cutoffs = cutoffs.sort_values("Wavelength_nm", ascending=False).reset_index(drop=True)

    cutoffs["nu_Hz"] = c / (cutoffs["Wavelength_nm"] * 1e-9)

    # scale nu so that it doens't fit to just high frequency number
    nu_scaled = cutoffs["nu_Hz"].to_numpy(dtype=float) / 1e14
    Vc = cutoffs["mean_Vc_V"].to_numpy(dtype=float)
    Vc_err = cutoffs["SEM_Vc_V"].to_numpy(dtype=float)

    popt, pcov = curve_fit(
        linear,
        nu_scaled,
        Vc,
        sigma=Vc_err,
        absolute_sigma=True,
    )

    m_scaled, b = popt
    m_scaled_err, b_err = np.sqrt(np.diag(pcov))

    Vc_fit = linear(nu_scaled, *popt)
    residuals = (Vc - Vc_fit) / Vc_err

    chi2 = np.sum(residuals**2)
    dof = len(Vc) - len(popt)
    reduced_chi2 = chi2 / dof

    # undo 1e14 scaling applied to frequency
    m = m_scaled / 1e14
    m_err = m_scaled_err / 1e14

    # Vc = (h/e) nu - phi/e
    h_exp = e * m
    h_err = e * m_err

    phi_J = -e * b
    phi_J_err = e * b_err

    return {
        "method": method_name,
        "cutoffs": cutoffs,
        "nu_scaled": nu_scaled,
        "Vc": Vc,
        "Vc_err": Vc_err,
        "popt": popt,
        "pcov": pcov,
        "m_V_s": m,
        "m_err_V_s": m_err,
        "b_V": b,
        "b_err_V": b_err,
        "h_J_s": h_exp,
        "h_err_J_s": h_err,
        "phi_J": phi_J,
        "phi_err_J": phi_J_err,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
    }

def plot_one_method(result):
    method_name = result["method"]
    cutoffs = result["cutoffs"]
    popt = result["popt"]

    nu_scaled = result["nu_scaled"]
    nu_plot = np.linspace(nu_scaled.min(), nu_scaled.max(), 500)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.errorbar(
        cutoffs["nu_Hz"],
        cutoffs["mean_Vc_V"],
        yerr=cutoffs["SEM_Vc_V"],
        fmt="o",
        capsize=4,
        color="midnightblue",
        ecolor="midnightblue",
        label="Mean cutoff voltage"
    )

    ax.plot(
        nu_plot * 1e14,
        linear(nu_plot, *popt),
        linewidth=2,
        color=colors[method_name],
        label="Weighted linear fit",
    )

    fit_text = (
        fr"$\chi^2 = {result['chi2']:.2f}$" "\n"
        fr"dof $= {result['dof']}$" "\n"
        fr"$\chi^2/\mathrm{{dof}} = {result['reduced_chi2']:.2f}$"
    )

    ax.text(
        0.05,
        0.95,
        fit_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.7",
            alpha=0.9,
        ),
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Mean cutoff voltage $V_c$ (V)")
    ax.set_title(f"{method_name}: cutoff voltage vs. frequency")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()

    fixed_name = (
        method_name.lower()
        .replace(":", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
    fig.savefig(figure_dir / f"{fixed_name}_vc_vs_frequency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_method_comparison(results):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for result in results:
        method_name = result["method"]
        cutoffs = result["cutoffs"]
        nu_scaled = result["nu_scaled"]
        nu_plot = np.linspace(nu_scaled.min(), nu_scaled.max(), 500)

        ax.errorbar(
            cutoffs["nu_Hz"],
            cutoffs["mean_Vc_V"],
            yerr=cutoffs["SEM_Vc_V"],
            fmt="o",
            capsize=4,
            color=data_colors[method_name],
            ecolor=data_colors[method_name],
            label=f"{method_name} data",
        )

        ax.plot(
            nu_plot * 1e14,
            linear(nu_plot, *result["popt"]),
            linewidth=3,
            color=colors[method_name],
            label=fr"{method_name} fit, $\chi^2$/dof = {result["reduced_chi2"]:.2f}"
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Mean cutoff voltage $\bar{V}_c$ (V)")
    ax.set_title("Comparison of cutoff analysis methods")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(figure_dir / "method1_vs_method2_vc_vs_frequency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    results = []

    for method_name, summary_file in method_files.items():
        result = fit_planck(summary_file, method_name)
        results.append(result)

        print(f"\n{method_name}")
        print("-" * len(method_name))
        print("Fit statistics:")
        print(f"chi^2 = {result['chi2']:.3f}")
        print(f"dof = {result['dof']}")
        print(f"chi^2/dof = {result['reduced_chi2']:.3f}")
        print("Slope and intercept:")
        print(f"m = {result['m_V_s']:.4e} +/- {result['m_err_V_s']:.4e} V s")
        print(f"b = {result['b_V']:.4f} +/- {result['b_err_V']:.4f} V")
        print("h and phi:")
        print(f"h = {result['h_J_s']:.8e} +/- {result['h_err_J_s']:.8e} J s")
        print(f"phi = {result['phi_J']:.8e} +/- {result['phi_err_J']:.8e} J")

        plot_one_method(result)

    plot_method_comparison(results)

    rows = []
    for result in results:
        rows.append({
            "Method": result["method"],
            "slope_V_s": result["m_V_s"],
            "slope_err_V_s": result["m_err_V_s"],
            "intercept_V": result["b_V"],
            "intercept_err_V": result["b_err_V"],
            "h_J_s": result["h_J_s"],
            "h_stat_err_J_s": result["h_err_J_s"],
            "phi_J": result["phi_J"],
            "phi_stat_err_J": result["phi_err_J"],
            "chi2": result["chi2"],
            "dof": result["dof"],
            "reduced_chi2": result["reduced_chi2"]
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(results_file, index=False)

    if len(results) == 2:
        method1, method2 = results
        delta_h = abs(method1["h_J_s"] - method2["h_J_s"])
        delta_phi_J = abs(method1["phi_J"] - method2["phi_J"])

        print("\nDifference between cutoff analysis methods")
        print(f"|h1 - h2| = {delta_h:.8e} J s")
        print(f"|phi1 - phi2| = {delta_phi_J:.8e} J")

    print(f"\nSaved Planck-fit summary to: {results_file}")
    print(f"Saved Planck-fit figures to: {figure_dir}")

if __name__ == "__main__":
    main()
