import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_file = "/home/rck67/jlab/photoelectric/data/current_statistics.csv"
figure_dir = "/home/rck67/jlab/photoelectric/figures/mean_current_with_sd.png"

stats = pd.read_csv(data_file)

wavelengths = [546.1, 435.7, 365]

# plot mean current with std error bars
plt.style.use("dark_background")

# choose zoom windows near the cutoff region of each wavelength
x_zoom = {
    546.1: (3.8, 5.8),
    435.7: (4.8, 6.2),
    365.0: (5.4, 6.8)
}

y_zoom = (-0.1, 1.2)

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 8),
    sharey="row"
)

for col, wavelength in enumerate(wavelengths):

    wavelength_stats = stats[stats["Wavelength_nm"] == wavelength]

    # top row is full plot
    ax = axes[0, col]

    ax.scatter(
        wavelength_stats["Voltage_V"],
        wavelength_stats["mean_I_nA"],
        color="lightsteelblue",
        s=15,
        zorder=3
    )

    ax.errorbar(
        wavelength_stats["Voltage_V"],
        wavelength_stats["mean_I_nA"],
        yerr=wavelength_stats["std_I_nA"],
        fmt="none",
        ecolor="white",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=2
    )

    ax.axhline(0, color="white", linestyle="--", linewidth=0.8)
    ax.set_title(f"{wavelength} nm")
    ax.set_xlabel("Retarding Voltage $V_r$ (V)")

    # bottom row is zoom near cutoff voltage
    ax_zoom = axes[1, col]

    ax_zoom.scatter(
        wavelength_stats["Voltage_V"],
        wavelength_stats["mean_I_nA"],
        color="lightsteelblue",
        s=20,
        zorder=3
    )

    ax_zoom.errorbar(
        wavelength_stats["Voltage_V"],
        wavelength_stats["mean_I_nA"],
        yerr=wavelength_stats["std_I_nA"],
        fmt="none",
        ecolor="white",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=2
    )

    ax_zoom.axhline(0, color="white", linestyle="--", linewidth=0.8)
    ax_zoom.set_xlim(x_zoom[wavelength])
    ax_zoom.set_ylim(y_zoom)
    ax_zoom.set_xlabel("Retarding Voltage $V_r$ (V)")
    ax_zoom.set_title(f"{wavelength} nm cutoff region")

# Shared y-labels
axes[0, 0].set_ylabel("Mean Photocurrent $\\bar{I}$ (nA)")
axes[1, 0].set_ylabel("Mean Photocurrent $\\bar{I}$ (nA)")

fig.tight_layout()

fig.savefig(
    "/home/rck67/jlab/photoelectric/figures/mean_current_with_sd_and_cutoff_zoom.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
