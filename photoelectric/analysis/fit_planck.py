import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.constants import c, e

output_dir = Path("/home/rck67/jlab/photoelectric/")
figure_dir = output_dir / "figures/method1"

method1_cutoffs = pd.read_csv("/home/rck67/jlab/photoelectric/data/method1_dubridge_cutoff_summary.csv")

# convert wavelength to frequencies
method1_cutoffs["nu_Hz"] = c / (method1_cutoffs["Wavelength_nm"] * 1e-9)

# find h value
nu_scaled = method1_cutoffs["nu_Hz"].to_numpy() / 1e14
Vc = method1_cutoffs["mean_Vc_V"].to_numpy()
Vc_err = method1_cutoffs["SEM_Vc_V"].to_numpy()

def linear(x, m, b):
    return m*x + b

popt, pcov = curve_fit(
    linear,
    nu_scaled,
    Vc,
    sigma=Vc_err,
    absolute_sigma=True
)

m_scaled, b = popt
m_scaled_err, b_err = np.sqrt(np.diag(pcov))

Vc_fit = linear(nu_scaled, *popt)
residuals = (Vc - Vc_fit) / Vc_err

chi2 = np.sum(residuals**2)
dof = len(Vc) - len(popt)

reduced_chi2 = chi2 / dof

print(f"chi^2 = {chi2:.3f}")
print(f"dof = {dof}")
print(f"chi^2/dof = {reduced_chi2:.3f}")

m = m_scaled / 1e14
m_err = m_scaled_err / 1e14

# planck's constant from slope
# m = h/e
h_exp = e * m
h_err = e * m_err

# work function from the intercept
# b = -phi/e
phi_exp = -e * b
phi_err = e * b_err

print("Slope and intercept:")
print(f"m = {m:.4e} +/- {m_err:.4e} V s")
print(f"b = {b:.4f} +/- {b_err:.4f} V")

print("Planck's constant and work function:")
print(f"h = {h_exp:.8e} +/- {h_err:.8e} J s")
print(f"phi = {phi_exp:.8e} +/- {phi_err:.8e} J")

nu_plot = np.linspace(
    nu_scaled.min(),
    nu_scaled.max(),
    500
)

plt.errorbar(
    method1_cutoffs["nu_Hz"],
    method1_cutoffs["mean_Vc_V"],
    yerr=method1_cutoffs["SEM_Vc_V"],
    color="midnightblue",
    ecolor="midnightblue",
    fmt="o",
    capsize=4
)

plt.plot(
    nu_plot * 1e14,
    linear(nu_plot, *popt),
    label="Weighted linear fit",
    color="cornflowerblue"
)

fit_text = (
    fr"$\chi^2 = {chi2:.2f}$"
    "\n"
    fr"$\chi^2/\mathrm{{dof}} = {reduced_chi2:.2f}$"
)

plt.gca().text(
    0.05,
    0.95,
    fit_text,
    transform=plt.gca().transAxes,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.9
    )
)

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Mean Cutoff Voltage $V_c$ (V)")
plt.title("Cutoff Voltage vs. Frequency")
plt.legend()
plt.tight_layout()

plt.savefig("/home/rck67/jlab/photoelectric/figures/method1_dubridge/vc_vs_frequency.png")
print("Saved Vc vs. Frequency plot.")