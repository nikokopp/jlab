import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns

from pathlib import Path
import re

from spe_analyzer import read_spe, parse_run_name, roi_stats

data_root = Path("/home/rck67/jlab/relativistic_dynamics/data/raw/calibration")
data_file = Path(data_root / "2026-10-09/20261009_calibration.Spe")
figure_dir = Path("/home/rck67/jlab/relativistic_dynamics/figures/calibration")
output_dir = Path("/home/rck67/jlab/relativistic_dynamics/data/processed/calibration_results")
figure_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# visually determine what features to identify in spectrum

# def compton_edge(E_gamma):
#     """
#     Calculate Compton edge from gamma ray energy.
#     """
#     m_e_c2 = 510.99895  # electron rest energy in keV
#     E_gamma = np.asarray(E_gamma)
#     E_c = E_gamma / (1 + m_e_c2/(2 * E_gamma))
#     return E_c

# gamma_energies = [30.85, 35.10, 53.16, 79.62, 81.00, 160.61, 223.25, 276.40, 302.85, 356.02, 383.85]
# compton_edges = compton_edge(gamma_energies)

spectrum = read_spe(data_file)

channels = spectrum["channels"]
counts = spectrum["counts"]
live_time = spectrum["live_time_s"]

channel_81_est = 145

# print(f"Number of channels: {len(channels)}")
# print(f"Total counts: {np.sum(counts)}")
# print(f"Live time: {live_time:.1f} s")

# fig, ax1 = plt.subplots(figsize=(10, 7))

# ax1.step(channels, counts, where="mid", linewidth=1, color="midnightblue")
# ax1.set(
#     xlabel="Channel",
#     ylabel="Counts per channel",
#     title="Ba-133 overnight calibration",
#     xlim=(0, 750),
#     yscale="log"
# )

# fig.tight_layout()
# fig.savefig(figure_dir / "ba133_overnight_calibration_raw_spectrum.png")
# plt.show()

# gaussian with linear background
def gaussian_linear(n, A, mu, sigma, B0, B1):
    gaussian = A * np.exp(-0.5 * ((n - mu) / sigma)**2)
    background = B0 + B1 * (n - channel_81_est)
    return gaussian + background

# gaussian with quadratic background
def gaussian_quadratic(n, A, mu, sigma, B0, B1, B2):
    gaussian = A * np.exp(-0.5 * ((n - mu) / sigma)**2)
    background = B0 + B1 * (n - channel_81_est) + B2 * (n - channel_81_est)**2
    return gaussian + background

# 79.62 and 81.00 keV double gaussian
E_79 = 79.62
E_81 = 81.00

intensity_ratio = 2.62 / 34.06

# provisional calibration slope (keV/channel)
slope_guess = 0.56
delta_ch = (E_81 - E_79) / slope_guess


def gaussian_doublet_quadratic(n, A, mu, sigma, B0, B1, B2):
    # 81.00 keV peak
    peak_81 = A * np.exp(-0.5 * ((n - mu) / sigma)**2)

    # 79.62 keV peak
    peak_79 = intensity_ratio * A * np.exp(-0.5 * ((n - (mu - delta_ch)) / sigma)**2)

    # quadratic background
    background = (B0 + B1 * (n - channel_81_est) + B2 * (n - channel_81_est)**2)

    return peak_81 + peak_79 + background


fit_min, fit_max = 130, 160
mask = (channels >= fit_min) & (channels <= fit_max)

x = channels[mask]
y = counts[mask]

y_err = np.sqrt(y)
p0_quadratic = [9700, 144.4, 4.1, 5500, -120, 2]

popt, pcov = curve_fit(
    gaussian_quadratic,
    x, y,
    p0=p0_quadratic,
    sigma=y_err,
    absolute_sigma=True
)

# save quadratic results
popt_quad = popt.copy()
pcov_quad = pcov.copy()

# initialize doublet using single gaussian results
p0_doublet = popt_quad.copy()

# estimate 81 keV peak centroid
p0_doublet[1] += (intensity_ratio / (1 + intensity_ratio)) * delta_ch

# adjust amplitude for the second peak
p0_doublet[0] /= (1 + intensity_ratio)

# fit double gaussian + quadratic background
popt_doublet, pcov_doublet = curve_fit(
    gaussian_doublet_quadratic,
    x,
    y,
    p0=p0_doublet,
    sigma=y_err,
    absolute_sigma=True
)

# resulting parameters and errors
A, mu_81, sigma, B0, B1, B2 = popt_doublet
perr_doublet = np.sqrt(np.diag(pcov_doublet))

mu_81_err = perr_doublet[1]

# fit
res_doublet = y - gaussian_doublet_quadratic(x, *popt_doublet)
norm_res = res_doublet / y_err

chi2_doublet = np.sum(norm_res**2)
dof_doublet = len(x) - len(popt_doublet)

print(f"81 keV centroid: {mu_81:.4f} +/- {mu_81_err:.4f}")
print(f"Reduced chi-squared: {chi2_doublet/dof_doublet:.3f}")

fig, (ax, ax_res) = plt.subplots(
    2, 1,
    figsize=(7.2, 5.1),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
)

# raw spectrum
plot_mask = (channels >= 120) & (channels <= 170)

ax.step(
    channels[plot_mask],
    counts[plot_mask],
    where="mid",
    color="0.65",
    linewidth=1,
    label="Raw spectrum"
)

# measurements with poisson error bars in fit region
ax.errorbar(
    x, y,
    yerr=y_err,
    fmt="o",
    color="black",
    markersize=3,
    linewidth=0.8,
    capsize=0,
    label="Data in fit region"
)

# fitted model inside fit interval
x_fit = np.linspace(fit_min, fit_max, 500)

ax.plot(
    x_fit,
    gaussian_doublet_quadratic(x_fit, *popt_doublet),
    color="cornflowerblue",
    linewidth=2,
    label="Double Gaussian"
)

# quadratic background
background = (
    B0
    + B1 * (x_fit - channel_81_est)
    + B2 * (x_fit - channel_81_est)**2
)

ax.plot(
    x_fit,
    background,
    "--",
    color="mediumorchid",
    linewidth=1.5,
    label="Quadratic background"
)

# shade fitting window
ax.axvspan(
    fit_min, fit_max,
    color="gray",
    alpha=0.10
)

ax.set_ylabel("Counts per channel")
ax.legend(loc="upper right", fontsize=9, frameon=False)

# fit statistics text
fit_text = (
    fr"$\mu_{{81 \mathrm{{keV}}}} = {mu_81:.2f} \pm {mu_81_err:.2f}$ channels"
    "\n"
    fr"$\chi^2/\mathrm{{dof}} = {chi2_doublet/dof_doublet:.2f}$"
)

ax.text(
    0.02, 0.97,
    fit_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="0.7",
        linewidth=0.8,
        alpha=0.95
    )
)

# normalized residuals
ax_res.errorbar(
    x,
    norm_res,
    yerr=np.ones_like(norm_res),
    fmt="o",
    color="black",
    markersize=3,
    linewidth=0.8,
    capsize=0
)

ax_res.axhline(
    0, color="gray",
    linestyle="--",
    linewidth=1
)

ax_res.set_xlabel("MCA channel")
ax_res.set_ylabel(r"$(N-M)/\sqrt{N}$")
ax_res.set_xlim(120, 170)
ax_res.set_ylim(-4.5, 4.5)

fig.subplots_adjust(
    left=0.13, right=0.98,
    bottom=0.12, top=0.98
)

fig.savefig(figure_dir / "ba133_81keV_doublet_fit.png")
plt.show()

# window uncertainty analysis
fit_windows = [
    (132, 158),
    (130, 160),
    (128, 162),
    (126, 164)
]

window_results = []

for fit_min_i, fit_max_i in fit_windows:
    mask = (channels >= fit_min_i) & (channels <= fit_max_i)

    x_i = channels[mask]
    y_i = counts[mask]
    y_err_i = np.sqrt(y_i)

    # fit using previous doublet parameters as initial guesses
    popt_i, pcov_i = curve_fit(
        gaussian_doublet_quadratic,
        x_i,
        y_i,
        p0=popt_doublet,
        sigma=y_err_i,
        absolute_sigma=True,
        maxfev=20000
    )

    perr_i = np.sqrt(np.diag(pcov_i))

    # goodness of fit
    residuals_i = y_i - gaussian_doublet_quadratic(x_i, *popt_i)

    chi2_i = np.sum((residuals_i / y_err_i)**2)
    dof_i = len(x_i) - len(popt_i)

    window_results.append({
        "Window": f"{fit_min_i}-{fit_max_i}",
        "Centroid": popt_i[1],
        "Centroid_err": perr_i[1],
        "Width": popt_i[2],
        "Reduced_chi2": chi2_i / dof_i
    })

window_df = pd.DataFrame(window_results)

mu_reference = popt_doublet[1]
window_df["Centroid_shift"] = window_df["Centroid"] - mu_reference

print(window_df.to_string(index=False))