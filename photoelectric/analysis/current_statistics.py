import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_file = "/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv"

data = pd.read_csv(data_file)

wavelengths = [546.1, 435.7, 365]

# folder to save data/plots
output_file = Path("/home/rck67/jlab/photoelectric/data/current_statistics.csv")
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/current_statistics")

# average repeated measurements within the same run
per_run = (
    data
    .groupby(
        ["Wavelength_nm", "Voltage_V", "Run"], 
        as_index=False
    )
    .agg(
        Current_nA=("Current_nA", "mean"),
        n_measurements=("Current_nA", "count")
    )
)

# plots using seaborn
plt.style.use("dark_background")

g = sns.relplot(
    data=per_run,
    x="Voltage_V",
    y="Current_nA",
    col="Wavelength_nm",
    kind="line",

    estimator="mean",
    errorbar="sd",
    err_style="bars",
    err_kws={
        "ecolor": "white",
        "capsize": 4,
        "elinewidth": 1.5,
        "capthick": 1.5
    },

    marker="o",
    markersize=5,
    color="lightsteelblue",

    height=4,
    aspect=1.2
)

g.set_axis_labels(
    "Retarding Voltage $V_r$ (V)",
    "Photocurrent $I$ (nA)"
)

g.set_titles("{col_name} nm")

plt.show()

# calculate statistics across indep runs
stats = (
    per_run
    .groupby(
        ["Wavelength_nm", "Voltage_V"],
        as_index=False
    )
    .agg(
        N=("Current_nA", "count"),
        mean_I_nA=("Current_nA", "mean"),
        variance_I_nA2=("Current_nA", "var"),
        std_I_nA=("Current_nA", "std")
    )
)

# SEM
stats["SEM_I_nA"] = stats["std_I_nA"] / np.sqrt(stats["N"])

# keep only voltages with multiple runs
stats = stats[stats["N"] > 1]

# print results
for wavelength in wavelengths:
    wavelength_stats = stats[stats["Wavelength_nm"] == wavelength]
    print(f"\n{wavelength} nm")
    print(wavelength_stats.to_string(index=False))

# save results
stats.to_csv(output_file, index=False)

