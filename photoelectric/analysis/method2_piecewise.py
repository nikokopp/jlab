import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
output_file = Path("/home/rck67/jlab/photoelectric/data/method2_piecewise_cutoffs_by_run.csv")
summary_file = Path("/home/rck67/jlab/photoelectric/data/method2_piecewise_cutoff_summary.csv")
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/method2_piecewise")
figure_dir.mkdir(parents=True, exist_ok=True)

wavelengths = [546.1, 435.7, 365.0]

fit_windows = {
    546.1: (4.7, 5.5),
    435.7: (5.2, 6.0),
    365.0: (5.7, 6.5),
}

sns.set_theme(style="ticks", context="paper")

def piecewise_linear_model(V, Vc, Ic, m_left, m_right):
    """
    Linear picewise with slope for left line and slope for right line and meeting point at Vc.

    V < Vc: I = Ic + m_left(V - Vc)
    V >= I = Ic + m_right(V - Vc)
    """
    x = V - Vc
    model = np.where(V < Vc, Ic + m_left*x, Ic + m_right*x)
    return model

def prepare_run_data(filename):
    data = pd.read_csv(filename)

    # if there are multiple measurements of the same voltage on the same run, average them
    # should only be the case for 4.5V on Run 1 of 546.1nm
    per_run = (
        data
        .groupby(["Wavelength_nm", "Run", "Voltage_V"], as_index=False)
        .agg(Current_nA=("Current_nA", "mean"))
    )

    current_stats = (
        per_run
        .groupby(["Wavelength_nm", "Voltage_V"], as_index=False)
        .agg(
            N_runs=("Current_nA", "count"),
            std_I_nA=("Current_nA", "std")
        )
    )

    # make sure there is not zero std
    # all currents recorded to accuracy of 0.001 nA
    # uniform rounding interval of width d has std = d/sqrt(12)
    current_resolution_nA = 0.001
    sigma_resolution_nA = current_resolution_nA / np.sqrt(12)

    current_stats["sigma_I_nA"] = np.maximum(
        current_stats["std_I_nA"].fillna(0.0),
        sigma_resolution_nA
    )

    per_run = per_run.merge(
        current_stats[["Wavelength_nm", "Voltage_V", "sigma_I_nA"]],
        on=["Wavelength_nm", "Voltage_V"],
        how="left"
    )

    return per_run

def fit_piecewise_per_run(run_data, vmin, vmax):
    """Fit one run over one selected voltage window."""
    fit_data = (
        run_data[
            (run_data["Voltage_V"] >= vmin)
            & (run_data["Voltage_V"] <= vmax)
        ]
        .sort_values("Voltage_V")
        .copy()
    )

    V = fit_data["Voltage_V"].to_numpy(dtype=float)
    I = fit_data["Current_nA"].to_numpy(dtype=float)
    sigma_I = fit_data["sigma_I_nA"].to_numpy(dtype=float)

    if len(V) < 7:
        raise ValueError(f"Only {len(V)} points in range; need at least 7.")

    if np.any(~np.isfinite(sigma_I)) or np.any(sigma_I <= 0):
        raise ValueError("All current uncertainties must be finite and greater than zero.")

    Vc_lower = V[2] + 1e-6
    Vc_upper = V[-3] - 1e-6

    if Vc_lower >= Vc_upper:
        raise ValueError("Fit window does not allow enough points around Vc.")

    # initial guesses, fit line to low voltage and high voltage, then use intersection as initial Vc
    n_side = min(4, len(V) // 2)
    if n_side < 2:
        raise ValueError("Not enough points to initialize the two lines.")

    m_left0, b_left0 = np.polyfit(V[:n_side], I[:n_side], 1)
    m_right0, b_right0 = np.polyfit(V[-n_side:], I[-n_side:], 1)

    slope_difference = m_left0 - m_right0
    if abs(slope_difference) > 1e-12:
        Vc0 = (b_right0 - b_left0) / slope_difference
    else:
        Vc0 = V[np.argmin(np.abs(I))]

    Vc0 = np.clip(Vc0, Vc_lower + 1e-4, Vc_upper - 1e-4)

    I_left_at_Vc = m_left0 * Vc0 + b_left0
    I_right_at_Vc = m_right0 * Vc0 + b_right0
    Ic0 = 0.5 * (I_left_at_Vc + I_right_at_Vc)

    popt, pcov = curve_fit(
        piecewise_linear_model,
        V,
        I,
        p0=[Vc0, Ic0, m_left0, m_right0],
        sigma=sigma_I,
        absolute_sigma=True,
        bounds=([Vc_lower, -np.inf, -np.inf, -np.inf], [Vc_upper,  np.inf,  np.inf,  np.inf]),
        method="trf",
        maxfev=100000,
    )

    if not np.all(np.isfinite(pcov)):
        raise RuntimeError("Non-finite covariance matrix from piecewise fit.")

    Vc, Ic, m_left, m_right = popt
    Vc_err, Ic_err, m_left_err, m_right_err = np.sqrt(np.diag(pcov))

    I_fit = piecewise_linear_model(V, *popt)
    residuals = (I - I_fit) / sigma_I

    chi2 = np.sum(residuals**2)
    dof = len(V) - len(popt)
    reduced_chi2 = chi2 / dof

    return {
        "fit_data": fit_data,
        "V": V,
        "I": I,
        "sigma_I": sigma_I,
        "popt": popt,
        "Vc": Vc,
        "Vc_err": Vc_err,
        "Ic": Ic,
        "Ic_err": Ic_err,
        "m_left": m_left,
        "m_left_err": m_left_err,
        "m_right": m_right,
        "m_right_err": m_right_err,
        "residuals": residuals,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
        "N_points": len(V),
    }

def main():
    data = prepare_run_data(data_file)
    rows = []

    for wavelength in wavelengths:
        vmin, vmax = fit_windows[wavelength]
        wavelength_data = data[data["Wavelength_nm"] == wavelength]

        for run in sorted(wavelength_data["Run"].unique()):
            run_data = wavelength_data[wavelength_data["Run"] == run]
            result = fit_piecewise_per_run(run_data, vmin, vmax)

            rows.append({
                "Wavelength_nm": wavelength,
                "Run": run,
                "Vmin_V": vmin,
                "Vmax_V": vmax,
                "N_points": result["N_points"],
                "Vc_V": result["Vc"],
                "Vc_fit_err_V": result["Vc_err"],
                "Ic_nA": result["Ic"],
                "Ic_err_nA": result["Ic_err"],
                "m_left_nA_per_V": result["m_left"],
                "m_left_err_nA_per_V": result["m_left_err"],
                "m_right_nA_per_V": result["m_right"],
                "m_right_err_nA_per_V": result["m_right_err"],
                "chi2": result["chi2"],
                "dof": result["dof"],
                "reduced_chi2": result["reduced_chi2"]
            })

            V = result["V"]
            I = result["I"]
            sigma_I = result["sigma_I"]
            residuals = result["residuals"]

            V_plot = np.linspace(V.min(), V.max(), 1000)
            I_fit = piecewise_linear_model(V_plot, *result["popt"])
            I_left = result["Ic"] + result["m_left"] * (V_plot - result["Vc"])
            I_right = result["Ic"] + result["m_right"] * (V_plot - result["Vc"])

            fig, (ax, ax_res) = plt.subplots(
                2,
                1,
                figsize=(7.5, 7.0),
                sharex=True,
                layout="constrained",
                gridspec_kw={"height_ratios": [3, 1]},
            )

            ax.errorbar(
                V,
                I,
                yerr=sigma_I,
                fmt="o",
                capsize=3,
                color="midnightblue",
                ecolor="midnightblue",
                label=f"Run {run} data"
            )

            palette = sns.color_palette("mako", 6)
            ax.plot(V_plot, I_fit, linewidth=2, color=palette[1], label="Piecewise linear fit")
            ax.plot(V_plot, I_left, "--", linewidth=1.5, color=palette[3], label="Low-voltage line")
            ax.plot(V_plot, I_right, "--", linewidth=1.5, color=palette[4], label="High-voltage line")
            ax.axvline(
                result["Vc"],
                linestyle=":",
                linewidth=1.8,
                color="black",
                label=fr"$V_c={result['Vc']:.4f}$ V",
            )
            ax.axhline(0, color="0.6", linewidth=1)

            ax.set_ylabel(r"Photocurrent $I$ (nA)")
            ax.set_title(fr"Method 2: {wavelength} nm, run {run}")

            fit_text = (
                fr"$V_c = {result['Vc']:.4f} \pm {result['Vc_err']:.4f}$ V" "\n"
                fr"$\chi^2 = {result['chi2']:.1f}$" "\n"
                fr"dof $= {result['dof']}$" "\n"
                fr"$\chi^2/\mathrm{{dof}} = {result['reduced_chi2']:.2f}$"
            )
            ax.text(
                0.03,
                0.96,
                fit_text,
                transform=ax.transAxes,
                va="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="0.8",
                ),
            )

            ax.legend(frameon=False)

            ax_res.scatter(V, residuals, color="midnightblue")
            ax_res.axhline(0, color="0.4", linewidth=1)
            ax_res.axhline(2, color="0.6", linestyle=":", linewidth=1)
            ax_res.axhline(-2, color="0.6", linestyle=":", linewidth=1)
            ax_res.set_xlabel(r"Retarding voltage $V_r$ (V)")
            ax_res.set_ylabel(r"Residual / $\sigma_I$")

            sns.despine(fig=fig)
            fig.savefig(
                figure_dir / f"fit_{wavelength}nm_run{run}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    results = pd.DataFrame(rows)
    results.to_csv(output_file, index=False)

    summary_rows = []
    for wavelength in wavelengths:
        d = results[results["Wavelength_nm"] == wavelength]
        n_runs = len(d)

        summary_rows.append({
            "Wavelength_nm": wavelength,
            "N_runs": n_runs,
            "mean_Vc_V": d["Vc_V"].mean(),
            "std_Vc_V": d["Vc_V"].std(ddof=1),
            "SEM_Vc_V": d["Vc_V"].std(ddof=1) / np.sqrt(n_runs),
            "fit_based_err_on_mean_V": np.sqrt(np.sum(d["Vc_fit_err_V"]**2)) / n_runs,
            "total_chi2": d["chi2"].sum(),
            "total_dof": d["dof"].sum(),
            "combined_reduced_chi2": d["chi2"].sum() / d["dof"].sum(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False)

    print(f"\nSaved run-by-run Method 2 results to: {output_file}")
    print(f"Saved Method 2 summary to: {summary_file}")
    print(f"Saved Method 2 figures to: {figure_dir}")
    print("\nMethod 2 summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
