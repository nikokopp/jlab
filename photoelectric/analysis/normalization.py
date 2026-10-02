import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit

data = pd.read_csv("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
stats = pd.read_csv("/home/rck67/jlab/photoelectric/data/current_statistics.csv")

wavelengths = [546.1, 435.7, 365]

# palette
plt.style.use("dark_background")
colors = sns.color_palette("husl", 5)

# normalize each run by current at V=0
zero_current = data[data["Voltage_V"] == 0.0][["Wavelength_nm", "Run", "Current_nA"]].copy()
zero_current = zero_current.rename(columns={"Current_nA": "I0_nA"})

data = data.merge(
    zero_current,
    on=["Wavelength_nm", "Run"],
    how="left",
    validate="many_to_one"
)

data["I_norm"] = data["Current_nA"] / data["I0_nA"]

for wavelength in wavelengths:
    wavelength_data = data[data["Wavelength_nm"] == wavelength]
    plt.figure(figsize=(7, 5))

    for run, color in zip(sorted(wavelength_data["Run"].unique()), colors):

        run_data = wavelength_data[
            wavelength_data["Run"] == run
        ].sort_values("Voltage_V")

        plt.scatter(
            run_data["Voltage_V"],
            run_data["I_norm"],
            color=color,
            marker="o",
            label=f"Run {run}"
        )

    plt.axhline(0, color="white", linewidth=0.8)

    plt.xlabel("Retarding Voltage (V)")
    plt.ylabel(r"Normalized Current $I/I(0)$")
    plt.title(f"{wavelength} nm")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"/home/rck67/jlab/photoelectric/figures/norm_currents/norm_current_{wavelength}nm.png", dpi=300)
    print(f"Saved plot to /figures/norm_currents/norm_current_{wavelength}nm.png")

# plot all 3 wavelengths against each other using mean and sd
norm_stats = (
    data
    .groupby(["Wavelength_nm", "Voltage_V"])["I_norm"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

plt.figure(figsize=(7, 5))

colors = sns.color_palette("husl", 5)[2:5]

for wavelength, color in zip(wavelengths, colors):
    wavelength_data = norm_stats[
        norm_stats["Wavelength_nm"] == wavelength
    ].sort_values("Voltage_V")

    plt.errorbar(
        wavelength_data["Voltage_V"],
        wavelength_data["mean"],
        yerr=wavelength_data["std"],
        marker="o",
        color=color,
        ecolor="white",
        markersize=4,
        capsize=3,
        label=f"{wavelength} nm"
    )

plt.axhline(0, color="white", linewidth=0.8)

plt.xlabel("Retarding Voltage (V)")
plt.ylabel(r"Mean Normalized Current $I/I(0)$")
plt.title("Mean Normalized Photoelectric Current")
plt.legend()
plt.tight_layout()

plt.savefig("/home/rck67/jlab/photoelectric/figures/norm_currents/mean_norm_currents_with_std.png", dpi=300)
print("Saved plot to /figures/norm_currents/mean_norm_currents_with_std.png")
