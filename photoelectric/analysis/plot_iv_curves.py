import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_file = "/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv"

data = pd.read_csv(data_file)

wavelengths = [546.1, 435.7, 365]

# folder to save plots
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/iv_curves")
figure_dir.mkdir(parents=True, exist_ok=True)

# palette
plt.style.use("dark_background")
colors = sns.color_palette("husl", 5)

for wavelength in wavelengths:
    # select wavelength
    wave_data = data[data["Wavelength_nm"] == wavelength]

    # make new fig
    plt.figure(figsize=(7,5))

    for (run, run_data), color in zip(wave_data.groupby("Run"), colors):
            plt.scatter(
            run_data["Voltage_V"],
            run_data["Current_nA"],
            color=color,
            marker="o",
            label=f"Run {run}"
        )

    plt.axhline(0, color="white", linestyle="--")

    plt.xlabel("Retarding Voltage $V_r$ (V)")
    plt.ylabel("Photocurrent $I$ (nA)")
    plt.title(f"$I$ vs. $V_r$ for {wavelength} nm")

    plt.legend()
    plt.tight_layout()

    plt.savefig(figure_dir / f"iv_{wavelength:g}nm.png", dpi=300)
    print(f"Saved plot to {figure_dir}/iv_{wavelength:g}nm.png")
