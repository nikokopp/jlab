import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.constants import c, e

output_dir = Path("/home/rck67/jlab/photoelectric/")
figure_dir = output_dir / "figures/method1"

method1_cutoffs = pd.read_csv("/home/rck67/jlab/photoelectric/data/method1_tanh_cutoff_stats.csv")

# convert wavelength to frequencies
method1_cutoffs["nu_Hz"] = c / (method1_cutoffs["Wavelength_nm"] * 1e-9)

plt.style.use("dark_background")

plt.errorbar(
    method1_cutoffs["nu_Hz"],
    method1_cutoffs["mean_Vc_V"],
    yerr=method1_cutoffs["SEM_Vc_V"],
    color="lightsteelblue",
    ecolor="white",
    fmt="o",
    capsize=4
)

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Mean Cutoff Voltage $V_c$ (V)")
plt.title("Cutoff Voltage vs. Frequency")

plt.tight_layout()
plt.savefig(figure_dir / "mean_vc_no_h.png")

# find h value
nu_scaled = method1_cutoffs["nu_Hz"].to_numpy() / 1e14
Vc = method1_cutoffs["mean_Vc_V"].to_numpy()
Vc_err = method1_cutoffs["SEM_Vc_V"].to_numpy()

def linear(x, m, b):
    return m*x + b

popt, pcov = curve_fit(
    linear,
    nu_scaled,
    Vc
)

m_scaled, b = popt
m_scaled_err, b_err = np.sqrt(np.diag(pcov))

m = m_scaled / 1e14
m_err = m_scaled_err / 1e14

h_exp = e * m
h_err = e * m_err

print(f"h = {h_exp:.4e} +/- {h_err:.4e} J s")

nu_plot = np.linspace(
    nu_scaled.min(),
    nu_scaled.max(),
    500
)

plt.plot(
    nu_plot * 1e14,
    linear(nu_plot, *popt),
    label="Unweighted linear fit"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel(r"Mean Cutoff Voltage $V_c$ (V)")
plt.title("Cutoff Voltage vs. Frequency")
plt.legend()
plt.tight_layout()
plt.show()