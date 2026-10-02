import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv"

data = pd.read_csv(data_file)

wavelengths = [546.1, 435.7, 365]

wave_data = data[data["Wavelength_nm"] == wavelengths[0]]

plt.style.use("dark_background")
colors = [
    "#CC6677",
    "#DDCC77",
    "#44AA99",
    "#88CCEE",
    "#AA4499",
]

for (run, run_data), color in zip(wave_data.groupby("Run"), colors):
    # measurements were not necessarily entered in voltage order, so sort for each run
    run_data = run_data.sort_values("Voltage_V")
    
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
plt.title(f"$I$ vs. $V_r$ for {wavelengths[0]} nm")

plt.legend()
plt.tight_layout()
plt.show()