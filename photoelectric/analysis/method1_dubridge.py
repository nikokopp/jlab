import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
output_file = Path("/home/rck67/jlab/photoelectric/data/method1_dubridge_cutoffs_by_run.csv")
summary_file = Path("/home/rck67/jlab/photoelectric/data/method1_dubridge_cutoff_summary.csv")
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/method1_dubridge")
figure_dir.mkdir(parents=True, exist_ok=True)

wavelengths = [546.1, 435.7, 365.0]

fit_windows = {
    546.1: (4.7, 5.5),
    435.7: (5.1, 6.0),
    365.0: (5.5, 6.5),
}

sns.set_theme(style="ticks", context="paper")

def dubridge_model(V, Vc, Ic, m, A):
    """
    Quadratic and linear model inspired by DuBridge (1933).
    Quadratic decay up until cutoff voltage, then linear fit afterwards.

    V < Vc: I = Ic + m(V - Vc) + A(V - Vc)^2
    V >= Vc: I = Ic + m(V - Vc)
    """
    x = V - Vc
    model = Ic + m*x + A*np.clip(Vc - V, 0, None)**2 # for V <= Vc, the quadratic term is zero
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

def fit_dubridge_per_run(run_data, vmin, vmax):
    """Fit one run of a wavelength over selected voltage window."""
    fit_data = (
        run_data[
            (run_data["Voltage_V"] >= vmin)
            & (run_data["Voltage_V"] <= vmax)
        ]
        .sort_values("Voltage_V").copy()
    )

    V = fit_data["Voltage_V"].to_numpy(dtype=float)
    I = fit_data["Current_nA"].to_numpy(dtype=float)
    sigma_I = fit_data["sigma_I_nA"].to_numpy(dtype=float)

    if len(V) < 7:
        raise ValueError(f"Only {len(V)} points in range, need at least 7.")

    # initial Vc is voltage with smallest current
    Vc0 = V[np.argmin(np.abs(I))]

    # initial linear estimate from last few voltage points
    n_tail = min(4, len(V))
    m0, b0 = np.polyfit(V[-n_tail:], I[-n_tail:], 1)
    Ic0 = m0*Vc0 + b0

    if np.any(V < Vc0):
        denom = (Vc0 - V[V < Vc0])**2
        excess = I[V < Vc0] - (Ic0 + m0*(V[V < Vc0] - Vc0))
        A_candidates = excess / denom
        A_candidates = A_candidates[
            np.isfinite(A_candidates) & (A_candidates > 0)
        ]
        A0 = np.median(A_candidates) if len(A_candidates) else 1.0
    else:
        A0 = 1.0

    # keep Vc inside fit window with at least 3 voltage measurements on either side
    Vc_lower = V[2] + 1e-6
    Vc_upper = V[-3] - 1e-6

    if Vc_lower >= Vc_upper:
        raise ValueError("Fit window does not allow enough points around Vc.")

    Vc0 = np.clip(Vc0, Vc_lower + 1e-4, Vc_upper - 1e-4)

    popt, pcov = curve_fit(
        dubridge_model,
        V,
        I,
        p0 = [Vc0, Ic0, m0, A0],
        sigma=sigma_I,
        absolute_sigma=True,
        bounds=([Vc_lower, -np.inf, -np.inf, 0.0], [Vc_upper, np.inf, np.inf, np.inf]),
        method="trf",
        maxfev=100000
    )

    Vc, Ic, m, A = popt
    Vc_err, Ic_err, m_err, A_err = np.sqrt(np.diag(pcov))

    I_fit = dubridge_model(V, *popt)
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
        "m": m,
        "m_err": m_err,
        "A": A,
        "A_err": A_err,
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
            result = fit_dubridge_per_run(run_data, vmin, vmax)

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
                "m_nA_per_V": result["m"],
                "m_err_nA_per_V": result["m_err"],
                "A_nA_per_V2": result["A"],
                "A_err_nA_per_V2": result["A_err"],
                "chi2": result["chi2"],
                "dof": result["dof"],
                "reduced_chi2": result["reduced_chi2"],
            })

            V = result["V"]
            I = result["I"]
            sigma_I = result["sigma_I"]
            residuals = result["residuals"]

            V_plot = np.linspace(V.min(), V.max(), 1000)
            I_fit = dubridge_model(V_plot, *result["popt"])
            I_quadratic = result["Ic"] + result["m"]*(V_plot - result["Vc"]) + result["A"]*(V_plot - result["Vc"])**2
            I_linear = result["Ic"] + result["m"]*(V_plot - result["Vc"])

            fig, (ax, ax_res) = plt.subplots(
                2,
                1,
                figsize=(7.5, 7.0),
                sharex=True,
                layout="constrained",
                gridspec_kw={"height_ratios": [3,1]}
            )

            ax.errorbar(
                V,
                I,
                yerr=sigma_I,
                fmt="o",
                capsize=3,
                color="midnightblue",
                label=f"Run {run} data",
            )

            palette = sns.color_palette("mako", 6)

            ax.plot(V_plot, I_fit, linewidth=2, label="Combined quadratic and linear fit", color=palette[1])
            ax.plot(V_plot, I_quadratic, "--", linewidth=2, label="Quadratic (V < Vc)", color=palette[3])
            ax.plot(V_plot, I_linear, "--", linewidth=2, label="Linear (V >= Vc)", color=palette[4])
            ax.axvline(
                result["Vc"],
                linestyle=":",
                linewidth=1.8,
                label=fr"$V_c={result['Vc']:.4f}$ V",
                color="black"
            )
            ax.axhline(0, color="0.6", linewidth=1)
            ax.set_ylabel(r"Photocurrent $I$ (nA)")
            ax.set_title(fr"Method 1: {wavelength} nm, run {run}")

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

    print(f"\nSaved run-by-run results to: {output_file}")
    print(f"Saved summary to: {summary_file}")
    print(f"Saved figures to: {figure_dir}")

if __name__ == "__main__":
    main()