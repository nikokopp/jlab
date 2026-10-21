"""Joint Poisson signal/background spectral fit for 8.13 Relativistic Dynamics.

Code prepared with AI assistance; verify with real spectra and record use in lab notes.

Place this script in analysis/ next to spe_analyzer.py and run it from the
relativistic_dynamics project root. Fits ONE nominal B and ONE acquisition day.

Model: counts[i, j] ~ Poisson(t[i] * [A(V[i]) * signal_pdf[j]
                                      + background_Hz * background_pdf[j]])
where A(V) comes from the previously fitted velocity-sweep trapezoid.

Assumptions:
  * signal and background channel distributions do not change with V;
  * the trapezoid fit used the SAME full MCA channel range as this script;
  * all spectra use the same MCA gain (one acquisition date);
  * the total background rate is approximately V-independent.

This is a plug-in spectral fit: trapezoid-fit uncertainty is NOT included.
The bootstrap below propagates Poisson counting uncertainty only.
"""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spe_analyzer import read_spe, parse_run_name


EPS = 1e-15


def transmission(V, fit_row):
    """Dimensionless asymmetric trapezoid, matching velocity_sweep.py."""
    v0 = float(fit_row["V_opt_kV"])
    half = 0.5 * float(fit_row["flat_width_kV"])
    rise_width = float(fit_row["rise_width_kV"])
    fall_width = float(fit_row["fall_width_kV"])
    v2, v3 = v0 - half, v0 + half
    rise = np.clip((V - (v2 - rise_width)) / rise_width, 0, 1)
    fall = np.clip(((v3 + fall_width) - V) / fall_width, 0, 1)
    return np.minimum(rise, fall)


def read_field_spectra(folder, B_target, date):
    rows = []
    for path in sorted(Path(folder).glob("*.Spe")):
        info = parse_run_name(path)
        if not info or "B_G" not in info:
            continue
        if not np.isclose(float(info["B_G"]), B_target, rtol=0, atol=1e-5):
            continue
        if info.get("date") and info["date"] != date:
            raise ValueError(
                f"Run {path.name} is dated {info['date']}, but --date is {date}. "
                "Fit different days separately to avoid mixing MCA gains."
            )
        if not info.get("date"):
            warnings.warn(f"Cannot verify measurement date from filename {path.name}. "
                          "Make sure --spectra-dir contains only the requested acquisition day.")
        sp = read_spe(path)
        rows.append((info["Vsel_kV"], sp))

    if len(rows) < 3:
        raise ValueError(f"Only {len(rows)} spectra found for B={B_target} G in {folder}; need >= 3")

    rows.sort(key=lambda x: x[0])
    channels = np.asarray(rows[0][1]["channels"], dtype=int)
    for V, sp in rows:
        if not np.array_equal(channels, sp["channels"]):
            raise ValueError(f"MCA channel arrays differ at V={V} kV")
        if float(sp["live_time_s"]) <= 0:
            raise ValueError("All live times must be positive")

    voltages = np.array([v for v, _ in rows], dtype=float)
    times = np.array([sp["live_time_s"] for _, sp in rows], dtype=float)
    counts = np.stack([sp["counts"] for _, sp in rows]).astype(np.int64)
    return channels, voltages, times, counts


def fit_templates(counts, signal_weights, initial=None, max_iter=2500, tol=1e-6):
    """Poisson mixture EM for TWO shared, unrestricted normalized MCA spectra.

    The total rates are held fixed by the voltage-sweep fit. Conditioning on
    the count total of each acquisition gives the same maximizer for the
    spectral shapes, since both shape PDFs sum to one.
    """
    y = np.asarray(counts, dtype=float)
    w = np.asarray(signal_weights, dtype=float)[:, None]
    n = y.shape[1]

    if initial is None:
        lo = int(np.argmin(w[:, 0]))
        hi = int(np.argmax(w[:, 0]))
        low_rate = y[lo] / max(y[lo].sum(), 1)
        high_rate = y[hi] / max(y[hi].sum(), 1)
        signal = np.maximum(high_rate - low_rate, 0) + EPS
        background = low_rate + EPS
        if signal.sum() <= n * 10 * EPS:
            signal = high_rate + EPS
    else:
        signal, background = (np.asarray(a, dtype=float).copy() for a in initial)
        signal += EPS
        background += EPS

    signal /= signal.sum()
    background /= background.sum()
    converged = False

    for iteration in range(max_iter):
        sig = w * signal[None, :]
        bkg = (1 - w) * background[None, :]
        prob_signal = sig / np.maximum(sig + bkg, EPS)

        signal_new = (y * prob_signal).sum(axis=0) + EPS
        background_new = (y * (1 - prob_signal)).sum(axis=0) + EPS
        signal_new /= signal_new.sum()
        background_new /= background_new.sum()

        change = max(np.abs(signal_new - signal).sum(),
                     np.abs(background_new - background).sum())
        signal, background = signal_new, background_new
        if change < tol:
            converged = True
            break

    return signal, background, converged, iteration + 1


def group_channels(channels, counts, bin_width):
    """Coarsen adjacent channel bins to stabilize signal/background extraction."""
    starts = np.arange(0, len(channels), bin_width)
    ends = np.r_[starts[1:], len(channels)]
    left = channels[starts].astype(float) - 0.5
    right = channels[ends - 1].astype(float) + 0.5
    centers = 0.5 * (left + right)
    grouped = np.add.reduceat(counts, starts, axis=1)
    return centers, left, right, grouped


def fractional_median_channel(left, right, pdf, minimum, maximum):
    """Median from grouped signal distribution with uniform within-bin density."""
    lo = np.maximum(left, minimum - 0.5)
    hi = np.minimum(right, maximum + 0.5)
    overlap = np.maximum(hi - lo, 0.0)
    fractions = overlap / (right - left)
    masses = pdf * fractions
    fraction_kept = float(masses.sum())
    if fraction_kept <= EPS:
        raise ValueError("No fitted electron signal inside chosen median-channel interval")
    cdf = np.cumsum(masses)
    target = 0.5 * fraction_kept
    k = min(int(np.searchsorted(cdf, target)), len(left) - 1)
    prior = float(cdf[k - 1]) if k else 0.0
    probability_density = pdf[k] / (right[k] - left[k])
    median = float(lo[k] + (target - prior) / max(probability_density, EPS))
    return median, fraction_kept


def load_calibration(date, reference_path, comparison_path):
    ref = json.loads(Path(reference_path).read_text())["linear_calibration"]
    a = float(ref["slope_keV_per_channel"])
    if date == "2026-10-09":
        b = float(ref["offset_keV"])
        cov = np.asarray(ref["covariance_keV"], dtype=float)
    else:
        comparison = pd.read_csv(comparison_path)
        match = comparison[comparison["date"].astype(str) == date]
        if len(match) != 1:
            raise ValueError(f"Expected exactly one fixed-slope calibration for {date}; found {len(match)}")
        row = match.iloc[0]
        b = float(row["fixed_slope_offset_keV"])
        cov = np.asarray(ast.literal_eval(row["fixed_slope_covariance_keV"]), dtype=float)
    return a, b, cov


def energy_from_channel(median, sigma_ch, a, b, covariance):
    energy = a * median + b
    gradient = np.array([median, 1.0])
    cal_sigma = np.sqrt(max(float(gradient @ covariance @ gradient), 0))
    statistical_sigma = abs(a) * sigma_ch
    total_sigma = np.hypot(statistical_sigma, cal_sigma)
    return energy, statistical_sigma, cal_sigma, total_sigma


def bootstrap_medians(counts, times, sig_rate, bg_rate, weight, edges_left, edges_right,
                      limits, signal, background, n_boot, seed):
    if n_boot <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    mu = times[:, None] * (sig_rate[:, None] * signal[None, :] +
                           bg_rate * background[None, :])
    samples = []
    for k in range(n_boot):
        synthetic = rng.poisson(mu)
        sig_boot, bg_boot, _ok, _niter = fit_templates(
            synthetic, weight, initial=(signal, background),
            max_iter=1400, tol=3e-6
        )
        med, _ = fractional_median_channel(edges_left, edges_right, sig_boot, *limits)
        samples.append(med)
    return np.asarray(samples)


def make_plots(out, channels, voltages, times, counts, expected, signal, background,
               V0, ch_median, limits):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(channels, signal, where="mid", label="Fitted electron signal PDF", color="midnightblue", lw=1.0)
    ax.step(channels, background, where="mid", label="Fitted background PDF", color="mediumpurple", lw=1.0)
    ax.axvline(ch_median, ls="--", c="black", label=f"Signal median = {ch_median:.2f}")
    ax.axvspan(limits[0], limits[1], color="gray", alpha=0.08, label="Median analysis interval")
    ax.set(xlabel="MCA channel", ylabel="Probability per fitted channel bin",
           title="Shared signal and background spectra estimated jointly")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "signal_background_templates.png", dpi=190)
    plt.close(fig)

    # Show data and model at representative voltages.
    picks = np.unique([0, len(voltages) // 2,
                       int(np.argmin(np.abs(voltages - V0))), len(voltages) - 1])
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.4 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, picks):
        observed_rate = counts[idx] / times[idx]
        fitted_rate = expected[idx] / times[idx]
        ax.step(channels, observed_rate, where="mid", color="0.35", lw=0.9, label="Observed")
        ax.plot(channels, fitted_rate, color="midnightblue", lw=1.3, label="Joint fit")
        ax.set_ylabel("Counts/s/bin")
        ax.set_title(f"V={voltages[idx]:.3f} kV; V - V0 = {voltages[idx] - V0:+.3f} kV", fontsize=10)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("MCA channel")
    fig.tight_layout()
    fig.savefig(out / "voltage_spectral_fits.png", dpi=190)
    plt.close(fig)

    resid = (counts - expected) / np.sqrt(np.maximum(expected, 1))
    fig, ax = plt.subplots(figsize=(10, max(3.2, len(voltages) * 0.48)))
    zlim = max(3.0, min(8.0, float(np.nanpercentile(np.abs(resid), 98))))
    im = ax.imshow(resid, aspect="auto", cmap="coolwarm", vmin=-zlim, vmax=zlim,
                   origin="lower", extent=[channels[0], channels[-1] + 1, -0.5, len(voltages)-0.5])
    ax.set_yticks(np.arange(len(voltages)))
    ax.set_yticklabels([f"{v:.3f}" for v in voltages], fontsize=8)
    ax.set(xlabel="MCA channel (grouped bins)",
           ylabel="Selector voltage (kV)", title="Poisson residuals: observed - predicted")
    fig.colorbar(im, ax=ax, label="(N - expectation) / sqrt(expectation)")
    fig.tight_layout()
    fig.savefig(out / "spectral_residuals.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-dir", required=True,
                        help="Directory holding B...G_Vsel...kV_r... .Spe files; searched recursively")
    parser.add_argument("--sweep-csv", default="data/processed/trapezoid_fits/velocity_sweep_results.csv")
    parser.add_argument("--reference", default="data/processed/calibration_results/overnight/overnight_calibration.json")
    parser.add_argument("--comparison", default="data/processed/calibration_results/comparison/fixed_slope_comparison.csv")
    parser.add_argument("--B", type=float, required=True, help="Nominal magnetic field in gauss (e.g., 100)")
    parser.add_argument("--date", required=True, help="Date of these electron spectra: YYYY-MM-DD")
    parser.add_argument("--bin-width", type=int, default=8, help="Adjacent MCA channels per fitted bin (default 8)")
    parser.add_argument("--median-min", type=int, default=0, help="Inclusive lower signal-median channel limit")
    parser.add_argument("--median-max", type=int, default=2047, help="Inclusive upper signal-median channel limit")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of Poisson resamples; 0 disables")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", default="data/processed/joint_energy_fit")
    args = parser.parse_args()

    out = Path(args.output) / f"B{args.B:g}_{args.date}"
    out.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv(args.sweep_csv)
    field_row = sweep[np.isclose(sweep["B_G"], args.B, atol=1e-5, rtol=0)]
    if len(field_row) != 1:
        raise ValueError(f"Expected 1 trapezoid fit for B={args.B}; found {len(field_row)}")
    field_row = field_row.iloc[0]
    channels_raw, voltages, times, counts_raw = read_field_spectra(args.spectra_dir, args.B, args.date)
    if args.bin_width < 1 or not (channels_raw[0] <= args.median_min < args.median_max <= channels_raw[-1]):
        raise ValueError("Require positive --bin-width and a valid median range within MCA channels")
    channels, edges_left, edges_right, counts = group_channels(
        channels_raw, counts_raw, args.bin_width
    )

    signal_rates = float(field_row["amplitude_Hz"]) * transmission(voltages, field_row)
    background_rate = float(field_row["background_Hz"])
    if background_rate <= 0 or np.max(signal_rates) <= 0:
        raise ValueError("The velocity sweep needs positive background and some nonzero signal")
    weights = signal_rates / (signal_rates + background_rate)
    print("IMPORTANT: this uses every MCA channel. The trapezoid fit must have used the same full channel range.")
    print(f"Fitting groups of {args.bin_width} MCA channels; {len(channels)} spectral bins.")
    print(f"B={args.B:g} G, date={args.date}, V0={field_row['V_opt_kV']:.4f} kV")
    print(f"Spectrum count={len(voltages)}, signal-fraction range: {weights.min():.3f} to {weights.max():.3f}")
    if np.ptp(weights) < 0.15:
        raise ValueError("The signal/background mixture barely changes across voltages: "
                         "cannot reliably separate two unknown spectra. Obtain off-peak data or a background template.")

    # Check that the fitted count-rate model is compatible with the spectra.
    predicted_total = (signal_rates + background_rate) * times
    obs_total = counts.sum(axis=1)
    total_z = (obs_total - predicted_total) / np.sqrt(np.maximum(predicted_total, 1))
    print("\nVoltage      Observed Hz  Predicted Hz   Total-rate z")
    for V, n, t, sr, z in zip(voltages, obs_total, times, signal_rates, total_z):
        print(f"{V:8.3f}      {n/t:9.3f}      {sr + background_rate:9.3f}      {z:+7.2f}")
    if (np.abs(total_z) > 3).any():
        warnings.warn("Some total rates differ >3 Poisson standard deviations from the sweep model. "
                      "Check the MCA ROI, duplicate runs, and trapezoid fit before relying on these results.")

    signal, background, converged, n_iter = fit_templates(counts, weights)
    if not converged:
        warnings.warn(f"EM did not meet convergence tolerance within {n_iter} steps")
    total_variation = 0.5 * float(np.abs(signal - background).sum())
    if total_variation < 0.1:
        warnings.warn("Fitted signal and background spectral shapes are very similar. "
                      "The electron median may be poorly constrained despite EM convergence.")

    limits = (args.median_min, args.median_max)
    n_med, retained = fractional_median_channel(edges_left, edges_right, signal, *limits)
    if retained < 0.90:
        warnings.warn(f"Only {retained:.1%} of inferred electron signal is in the chosen median interval. "
                      "Inspect the signal spectrum carefully; do not discard signal arbitrarily.")

    samples = bootstrap_medians(counts, times, signal_rates, background_rate,
                                weights, edges_left, edges_right, limits, signal, background,
                                args.bootstrap, args.seed)
    sigma_n = float(np.std(samples, ddof=1)) if len(samples) > 1 else float("nan")
    a, b, covariance = load_calibration(args.date, args.reference, args.comparison)
    K, sigma_stat, sigma_cal, sigma_total = energy_from_channel(
        n_med, sigma_n, a, b, covariance
    )

    # The independent horizontal coordinate for the eventual K-versus-beta plot.
    # These formulas follow the user's existing Gaussian-units momentum analysis.
    d_cm, sigma_d_cm = 0.180, 0.003
    volts_per_statvolt = 299.792458
    B_corrected = float(field_row["B_corrected_G"])
    B_error = float(field_row["Berr_corrected_G"])
    V0 = float(field_row["V_opt_kV"])
    V0_error = float(field_row["V_opt_stat_err_kV"])
    beta = (V0 * 1000.0 / volts_per_statvolt / d_cm) / B_corrected
    if not 0 < beta < 1:
        raise ValueError(f"Nonphysical beta={beta}: check E/B and unit conversions")
    gamma = 1 / np.sqrt(1 - beta**2)
    beta_err_meas = beta * np.hypot(V0_error/V0, B_error/B_corrected)
    beta_err_gap_sys = beta * (sigma_d_cm / d_cm)
    gamma_err_meas = gamma**3 * beta * beta_err_meas
    gamma_err_gap_sys = gamma**3 * beta * beta_err_gap_sys

    expected = times[:, None] * (signal_rates[:, None] * signal[None, :] +
                                 background_rate * background[None, :])
    make_plots(out, channels, voltages, times, counts, expected,
               signal, background, float(field_row["V_opt_kV"]), n_med, limits)

    pd.DataFrame({"channel_center": channels, "left_edge": edges_left, "right_edge": edges_right,
                  "signal_probability": signal,
                  "background_probability": background}).to_csv(out / "signal_background_templates.csv", index=False)
    pd.DataFrame({"voltage_kV": voltages, "live_time_s": times,
                  "observed_count_total": obs_total,
                  "expected_count_total": predicted_total,
                  "signal_rate_Hz": signal_rates,
                  "background_rate_Hz": background_rate,
                  "signal_fraction": weights,
                  "total_count_residual_z": total_z}).to_csv(out / "voltage_diagnostics.csv", index=False)
    if len(samples):
        pd.DataFrame({"median_channel": samples}).to_csv(out / "bootstrap_medians.csv", index=False)
    pd.DataFrame([{
        "B_G": args.B, "date": args.date, "V0_kV": float(field_row["V_opt_kV"]),
        "median_channel": n_med, "median_stat_err_ch": sigma_n,
        "signal_probability_in_median_interval": retained,
        "K_deposited_keV": K, "K_stat_err_keV": sigma_stat,
        "K_cal_err_keV": sigma_cal, "K_combined_err_keV": sigma_total,
        "beta": beta, "gamma": gamma,
        "beta_err_measurement_approx": beta_err_meas,
        "beta_err_gap_common_sys": beta_err_gap_sys,
        "gamma_err_measurement_approx": gamma_err_meas,
        "gamma_err_gap_common_sys": gamma_err_gap_sys,
        "B_corrected_G": B_corrected, "V0_stat_err_kV": V0_error,
        "spectra_count": len(voltages), "EM_converged": converged,
        "EM_iterations": n_iter,
    }]).to_csv(out / "energy_summary.csv", index=False)

    print(f"\nEM converged: {converged} ({n_iter} iterations)")
    print(f"Electron-signal probability in median window: {retained:.3f}")
    print(f"Signal/background template total-variation distance: {total_variation:.3f}")
    print(f"Signal median channel: {n_med:.2f} +/- {sigma_n:.2f} (Poisson stat)")
    print(f"Deposited energy: {K:.2f} +/- {sigma_stat:.2f} (stat) "
          f"+/- {sigma_cal:.2f} (calibration) keV")
    print(f"beta = E/B = {beta:.4f}; gamma = {gamma:.4f}")
    print("These errors do NOT include spectral-model, gain-drift, or trapezoid-fit uncertainty.")
    print(f"Files saved to: {out}")


if __name__ == "__main__":
    main()
