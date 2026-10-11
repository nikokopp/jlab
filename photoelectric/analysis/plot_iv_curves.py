import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_file = "/home/rck67/jlab/photoelectric/data/current_statistics.csv"
data = pd.read_csv(data_file)
# print(data.head())

wavelengths = [546.1, 435.7, 365.0]

palette = sns.color_palette("mako", 5)

# folder to save plots
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/iv_curves")

# stack 3 plots
fig, axes = plt.subplots(
    3, 1,
    figsize=(7,6),
    sharex=True
)

panel_labels = ["(a)", "(b)", "(c)"]

for ax, wavelength, panel_label in zip(axes, wavelengths, panel_labels):
    # select wavelength
    wave_data = data[data["Wavelength_nm"] == wavelength].sort_values("Voltage_V")

    # plot only data with repeated measurements so we can have error bars
    repeated = wave_data["std_I_nA"].notna()

    ax.errorbar(
        wave_data.loc[repeated, "Voltage_V"],
        wave_data.loc[repeated, "mean_I_nA"],
        yerr=wave_data.loc[repeated, "std_I_nA"],
        fmt="o",
        color=palette[1],
        ecolor=palette[3],
        markersize=4,
        capsize=2,
        linewidth=1
    )

    # line at 0 current
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    # wavelength label
    ax.text(
        0.97,
        0.90,
        rf"{panel_label} {wavelength} nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold"
    )

    ax.tick_params(direction="in")

fig.supylabel(r"Photocurrent $I$ (nA)")
axes[-1].set_xlabel(r"Retarding Voltage $V_r$ (V)")
axes[-1].set_xlim(-0.5, 8.5)

plt.tight_layout()

plt.savefig(figure_dir / "iv_curves_mean_stacked.png", dpi=300, bbox_inches="tight")
print(f"Saved plot to {figure_dir}/iv_curves_mean_stacked.png")

plt.show()