import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import xlogy

from pathlib import Path
from spe_analyzer import read_spe, parse_run_name, roi_stats

# gaussmeter calibration effects
def correct_magnetic_field(B_measured):
    mag = 99.69  # reference magnet
    sigma_mag = 0.01 * mag # +/- 1% uncertainty on reference magnet

    # positive and negative calibration readings on reference magnet
    calibrations = {
        70.0: (108.9, -106.8),
        80.0: (104.0, -102.2),
        89.4: (106.0, -103.6),
        91.9: (105.0, -105.4),
        100.0: (104.0, -102.2),
        110.0: (105.1, -106.2),
        120.0: (107.3, -107.4)
    }
    
    sigma_cal = 0.02
    
    B_positive, B_negative = calibrations[B_measured]

    # Gaussmeter zero offset
    z = (B_positive + B_negative) / 2

    # Gaussmeter scale factor
    s = (B_positive - B_negative) / (2 * mag)

    # uncertainties
    sigma_z = sigma_cal / np.sqrt(2)
    sigma_s = np.sqrt(
        (2 * sigma_cal**2) / (2 * mag)**2
        + (s * sigma_mag / mag)**2
    )

    B_corrected = (B_measured - z) / s

    Berr_calibration = np.sqrt(
        sigma_z**2 + (B_corrected * sigma_s)**2
    ) / np.abs(s)

    # uncertainty in the measured magnetic field (G)
    sigma_B_measured = 2.0

    # combine measurement and calibration uncertainties
    Berr_corrected = np.sqrt(
        Berr_calibration**2
        + (sigma_B_measured / np.abs(s))**2
    )

    return B_corrected, Berr_corrected

def asymmetric_trapezoid(V, bg, amplitude, V_opt, flat_halfwidth, rise_width, fall_width):

    V2 = V_opt - flat_halfwidth
    V3 = V_opt + flat_halfwidth

    # rising edge
    rise = np.clip(
        (V - (V2 - rise_width)) / rise_width, 0, 1
    )

    # falling edge
    fall = np.clip(
        ((V3 + fall_width) - V) / fall_width, 0, 1
    )

    transmission = np.minimum(rise, fall)

    return bg + amplitude * transmission


def plot_velocity_sweep(folder, figure_dir, B_target, channel_min, channel_max):
    folder = Path(folder)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    voltages = []
    rates = []
    rate_errors = []
    counts = []
    live_times = []

    B_corrected, Berr_corrected = correct_magnetic_field(B_target)

    for filename in folder.glob("*.Spe"):
        run_info = parse_run_name(filename)

        if run_info is None:
            continue

        if not np.isclose(run_info["B_G"], B_target):
            continue

        spectrum = read_spe(filename)

        stats = roi_stats(spectrum, channel_min, channel_max)

        voltages.append(run_info["Vsel_kV"])
        rates.append(stats["count_rate_Hz"])
        rate_errors.append(stats["count_rate_error_Hz"])
        counts.append(stats["total_counts"])
        live_times.append(spectrum["live_time_s"])

        print(
            f"Vsel = {run_info['Vsel_kV']} kV | "
            f"counts = {stats['total_counts']} | "
            f"rate = {stats['count_rate_Hz']} +/- "
            f"{stats['count_rate_error_Hz']} counts/s"
        )

    # convert to arrays and sort by voltage
    if len(voltages) <= 6:
        raise ValueError(
            f"Only {len(voltages)} measurements found for B = {B_target} G. "
            "Need more than six for this six-parameter fit."
        )

    voltages = np.asarray(voltages, dtype=float)
    rates = np.asarray(rates, dtype=float)
    rate_errors = np.asarray(rate_errors, dtype=float)
    counts = np.asarray(counts, dtype=int)
    live_times = np.asarray(live_times, dtype=float)

    order = np.argsort(voltages)

    voltages = voltages[order]
    rates = rates[order]
    rate_errors = rate_errors[order]
    counts = counts[order]
    live_times = live_times[order]

    # trapezoid fit
    fit = fit_trapezoid(voltages, counts, live_times)

    bg, amplitude, V_opt, flat_halfwidth, rise_width, fall_width = fit.x

    # parameter covariance and statistical uncertainty
    J = fit.jac

    if np.linalg.matrix_rank(J) == len(fit.x):
        pcov = np.linalg.inv(J.T @ J)
        V_opt_err = np.sqrt(max(pcov[2, 2], 0))
    else:
        V_opt_err = np.nan
        print("Warning: fit covariance is singular.")

    print(f"V_0 = {V_opt:.4f} +/- {V_opt_err:.4f} kV")

    # predicted counts
    expected_counts = (
        asymmetric_trapezoid(voltages, *fit.x) * live_times
    )
    expected_counts = np.clip(expected_counts, 1e-12, None)

    # goodness of fit
    chi2 = np.sum((counts - expected_counts)**2 / expected_counts)
    dof = len(counts) - len(fit.x)
    deviance = np.sum(fit.fun**2)
    poisson_residuals = -fit.fun

    # fit curve
    V_fit = np.linspace(voltages.min(), voltages.max(), 500)
    R_fit = asymmetric_trapezoid(V_fit, *fit.x)

    # plot data and fitted curve

    fig, (ax, ax_res) = plt.subplots(
        2, 1,
        figsize=(7.5, 6),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.5, 1],
            "hspace": 0.08
        }
    )

    # data
    data_handle = ax.errorbar(
        voltages,
        rates,
        yerr=rate_errors,
        fmt="o",
        color="#001969",
        ecolor="#001969",
        capsize=2,
        markersize=4,
        label="Measured count rate",
        zorder=3
    )

    # trapezoid fit
    fit_handle, = ax.plot(
        V_fit,
        R_fit,
        color="#7e5bac",
        linewidth=1.5,
        label="Trapezoid fit",
        zorder=2
    )

    # optimal voltage line
    opt_handle = ax.axvline(
        V_opt,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=(
            rf"$V_0 = {V_opt:.3f} "
            rf"\pm {V_opt_err:.3f}_{{\mathrm{{stat}}}}$ kV"
        )
    )

    # space for legend/stats
    ymin = min(
        np.min(rates - rate_errors),
        np.min(R_fit)
    )

    ymax = max(
        np.max(rates + rate_errors),
        np.max(R_fit)
    )

    yrange = ymax - ymin

    ax.set_ylim(
        ymin - 0.10 * yrange,
        ymax + 0.55 * yrange
    )

    # legend
    ax.legend(
        handles=[fit_handle, opt_handle, data_handle],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        borderpad=0.4,
        labelspacing=0.35
    )

    # statistics
    stats_text = (
        rf"$\chi^2/\mathrm{{dof}} = "
        rf"{chi2:.2f}/{dof} = {chi2/dof:.2f}$"
        "\n"
        rf"$D/\mathrm{{dof}} = "
        rf"{deviance:.2f}/{dof} = {deviance/dof:.2f}$"
    )

    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        linespacing=1.3,
        bbox=dict(
            facecolor="white",
            edgecolor="0.7",
            alpha=0.95,
            boxstyle="round,pad=0.35"
        )
    )

    ax.set_ylabel("Count rate (counts/s)")
    ax.set_title(
        f"Voltage sweep at B = {B_corrected:.2f} G",
        fontsize=12,
        pad=12
    )

    # Poisson deviance residuals
    ax_res.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax_res.plot(
        voltages,
        poisson_residuals,
        "o",
        markersize=4,
        color="#001969"
    )

    ax_res.set_xlabel("Velocity selector voltage (kV)")
    ax_res.set_ylabel("Deviance residual")

    # layout
    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        top=0.92,
        bottom=0.12,
        hspace=0.08
    )

    # save figure
    fig.savefig(
        figure_dir / f"velocity_sweep_B{B_target}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    pd.DataFrame([{
        "B_G": B_target,
        "B_corrected_G": B_corrected,
        "Berr_corrected_G": Berr_corrected,

        "V_opt_kV": V_opt,
        "V_opt_stat_err_kV": V_opt_err,

        "background_Hz": bg,
        "amplitude_Hz": amplitude,
        "rise_width_kV": rise_width,
        "fall_width_kV": fall_width,
        "flat_width_kV": 2 * flat_halfwidth,

        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": chi2 / dof,
        "poisson_deviance": deviance,
        "reduced_deviance": deviance / dof
    }]).to_csv(
        f"/home/rck67/jlab/relativistic_dynamics/data/processed/trapezoid_fits/trapezoid_fit_B{B_target}.csv",
        index=False
    )

def fit_trapezoid(voltages, counts, live_times):

    rates = counts / live_times
    span = np.ptp(voltages)

    # initial guesses for the six parameters
    p0 = [
        max(rates.min(), 0.001),
        max(np.ptp(rates), 0.001),
        voltages[np.argmax(rates)],
        0.05 * span,
        0.20 * span,
        0.20 * span
    ]

    # bounds on the fit parameters
    lower = [
        0, 0, voltages.min(), 0,
        0.01 * span, 0.01 * span
    ]

    upper = [
        np.inf, np.inf, voltages.max(),
        0.5 * span, span, span
    ]

    def residuals(params, observed_counts):
        predicted_rates = asymmetric_trapezoid(voltages, *params)

        expected_counts = np.clip(predicted_rates * live_times,  1e-12, None)

        deviance = 2 * (
            expected_counts - observed_counts
            + xlogy(
                observed_counts,
                observed_counts / expected_counts
            )
        )

        return np.sign(
            expected_counts - observed_counts
        ) * np.sqrt(np.maximum(deviance, 0))

    fit = least_squares(
        residuals,
        x0=p0,
        bounds=(lower, upper),
        args=(counts,),
        max_nfev=10000
    )

    return fit

def combine_velocity_sweep_results(folder_dir):
    folder_dir = Path(folder_dir)

    # find all individual trapezoid fit CSVs
    csv_files = sorted(folder_dir.glob("trapezoid_fit_B*.csv"))

    if not csv_files:
        raise FileNotFoundError("No trapezoid fit CSV files found.")

    # read and combine the CSVs
    data = pd.concat(
        [pd.read_csv(file) for file in csv_files],
        ignore_index=True
    )

    # sort by measured magnetic field
    data = data.sort_values("B_G").reset_index(drop=True)

    # save combined results
    output_file = folder_dir / "velocity_sweep_results.csv"
    data.to_csv(output_file, index=False)

    print(f"Combined {len(csv_files)} files into {output_file}")

    return data

# ======================================================================
# ======================================================================
#
#                   PAPER FIGURE
#              TRAPEZOID FIT AT 120 G
#
# ======================================================================
# ======================================================================

if __name__ == "__main__":

    root = Path("/home/rck67/jlab/relativistic_dynamics")

    raw_dir = root / "data/processed/velocity_sweep"
    results_file = (
        root / "data/processed/trapezoid_fits/velocity_sweep_results.csv"
    )
    paper_dir = root / "figures/paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    B_target = 120.0
    channel_min = 0
    channel_max = 2048

    # ------------------------------------------------------------------
    # Load fitted parameters for nominal B = 120 G
    # ------------------------------------------------------------------

    fit_table = pd.read_csv(results_file)

    selected = fit_table[
        np.isclose(fit_table["B_G"], B_target)
    ]

    if len(selected) != 1:
        raise ValueError("Expected exactly one fit for B = 120 G.")

    fit_row = selected.iloc[0]

    B_corrected = fit_row["B_corrected_G"]
    V_opt = fit_row["V_opt_kV"]
    V_opt_err = fit_row["V_opt_stat_err_kV"]

    fit_params = [
        fit_row["background_Hz"],
        fit_row["amplitude_Hz"],
        V_opt,
        fit_row["flat_width_kV"] / 2,
        fit_row["rise_width_kV"],
        fit_row["fall_width_kV"]
    ]

    # ------------------------------------------------------------------
    # Load original measurements at 120 G
    # ------------------------------------------------------------------

    records = []

    for filename in sorted(raw_dir.glob("*.Spe")):

        run_info = parse_run_name(filename)

        if run_info is None:
            continue

        if not np.isclose(run_info["B_G"], B_target):
            continue

        spectrum = read_spe(filename)
        stats = roi_stats(spectrum, channel_min, channel_max)

        records.append({
            "voltage": run_info["Vsel_kV"],
            "rate": stats["count_rate_Hz"],
            "rate_err": stats["count_rate_error_Hz"],
            "counts": stats["total_counts"],
            "live_time": spectrum["live_time_s"]
        })

    if len(records) <= 6:
        raise ValueError(
            f"Found only {len(records)} measurements at 120 G. "
            f"Check the raw .Spe directory: {raw_dir}"
        )

    measurements = pd.DataFrame(records).sort_values("voltage")

    voltages = measurements["voltage"].to_numpy(dtype=float)
    rates = measurements["rate"].to_numpy(dtype=float)
    rate_errors = measurements["rate_err"].to_numpy(dtype=float)
    counts = measurements["counts"].to_numpy(dtype=float)
    live_times = measurements["live_time"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Reconstruct fitted curve and deviance residuals
    # ------------------------------------------------------------------

    V_fit = np.linspace(
        voltages.min(),
        voltages.max(),
        600
    )

    R_fit = asymmetric_trapezoid(V_fit, *fit_params)

    expected_counts = np.clip(
        asymmetric_trapezoid(voltages, *fit_params) * live_times,
        1e-12,
        None
    )

    deviance_terms = 2 * (
        expected_counts - counts
        + xlogy(counts, counts / expected_counts)
    )

    poisson_residuals = (
        np.sign(counts - expected_counts)
        * np.sqrt(np.maximum(deviance_terms, 0))
    )

    chi2 = np.sum(
        (counts - expected_counts)**2 / expected_counts
    )
    deviance = np.sum(deviance_terms)
    dof = len(counts) - len(fit_params)

    # ==================================================================
    # FIGURE: TRAPEZOID FIT AND RESIDUALS
    # ==================================================================

    fig_paper, (ax, ax_res) = plt.subplots(
        2, 1,
        figsize=(8.5, 5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 0.75],
            "hspace": 0.07
        }
    )

    fig_paper.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.975,
        bottom=0.115,
        hspace=0.07
    )

    label_size = 12
    tick_size = 10
    legend_size = 10
    stats_size = 9.5

    # ------------------------------------------------------------------
    # Measured count rate
    # ------------------------------------------------------------------

    data_handle = ax.errorbar(
        voltages,
        rates,
        yerr=rate_errors,
        fmt="o",
        color="#001969",
        ecolor="#001969",
        capsize=3,
        markersize=4,
        elinewidth=1.2,
        label="Measured count rate",
        zorder=3
    )

    # ------------------------------------------------------------------
    # Trapezoid fit
    # ------------------------------------------------------------------

    fit_handle, = ax.plot(
        V_fit,
        R_fit,
        color="#7e5bac",
        linewidth=2,
        label="Trapezoid fit",
        zorder=2
    )

    # Optimal voltage
    opt_handle = ax.axvline(
        V_opt,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=(
            rf"$V_0 = {V_opt:.3f} "
            rf"\pm {V_opt_err:.3f}_{{\mathrm{{stat}}}}$ kV"
        )
    )

    # ------------------------------------------------------------------
    # Axes and formatting
    # ------------------------------------------------------------------

    ymin = min(
        np.min(rates - rate_errors),
        np.min(R_fit)
    )

    ymax = max(
        np.max(rates + rate_errors),
        np.max(R_fit)
    )

    yrange = ymax - ymin

    # Room at top for legend and statistics
    ax.set_ylim(
        ymin - 0.10 * yrange,
        ymax + 0.43 * yrange
    )

    ax.set_ylabel(
        "Count rate (counts/s)",
        fontsize=label_size
    )

    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.18)

    # Legend: upper-right
    ax.legend(
        handles=[fit_handle, opt_handle, data_handle],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        fontsize=legend_size,
        framealpha=0.95,
        borderpad=0.4,
        labelspacing=0.35
    )

    # ------------------------------------------------------------------
    # Fit statistics: upper-left
    # ------------------------------------------------------------------

    stats_text = (
        rf"$\chi^2/\mathrm{{dof}} = "
        rf"{chi2:.2f}/{dof} = {chi2/dof:.2f}$"
        "\n"
        rf"$D/\mathrm{{dof}} = "
        rf"{deviance:.2f}/{dof} = {deviance/dof:.2f}$"
    )

    ax.text(
        0.02, 0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=stats_size,
        linespacing=1.4,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.95,
            "boxstyle": "round,pad=0.4"
        }
    )

    # ------------------------------------------------------------------
    # Poisson deviance residuals
    # ------------------------------------------------------------------

    ax_res.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.2
    )

    ax_res.plot(
        voltages,
        poisson_residuals,
        "o",
        markersize=4,
        color="#001969"
    )

    ax_res.set_xlabel(
        "Velocity selector voltage (kV)",
        fontsize=label_size
    )

    ax_res.set_ylabel(
        "Deviance residual",
        fontsize=label_size
    )

    ax_res.tick_params(labelsize=tick_size)
    ax_res.grid(alpha=0.18)

    voltage_span = np.ptp(voltages)
    ax_res.set_xlim(
        voltages.min() - 0.02 * voltage_span,
        voltages.max() + 0.02 * voltage_span
    )

    # ------------------------------------------------------------------
    # Save and preview
    # ------------------------------------------------------------------

    pdf_file = paper_dir / "trapezoid_fit_120G_paper.pdf"
    png_file = paper_dir / "trapezoid_fit_120G_paper.png"

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

    print(f"Nominal magnetic field: {B_target:.1f} G")
    print(f"Corrected magnetic field: {B_corrected:.2f} G")
    print(f"Saved PDF: {pdf_file}")
    print(f"Saved PNG: {png_file}")

    plt.show()

