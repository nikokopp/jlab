import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit

data = pd.read_csv("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
stats = pd.read_csv("/home/rck67/jlab/photoelectric/data/current_statistics.csv")

wavelengths = [546.1, 435.7, 365]

# manually determined the minimum V for which data appears to just be reverse current
reverse_min_V = {
    365.0: 7.0,
    435.7: 6.0,
    546.1: 6.0,
}

# take statistics from reverse current region
reverse_stats = []

for wavelength in wavelengths:
    region = stats[
        (stats["Wavelength_nm"] == wavelength)
        & (stats["Voltage_V"] >= reverse_min_V[wavelength])
    ].copy()

    reverse_stats.append(region)

reverse_stats = pd.concat(reverse_stats, ignore_index=True)

# fit to mean reverse current in each wavelength separately

# define linear model to fit to reverse current
def linear(V, m, b):
    return m * V + b

# this will store slope, intercept, slope error, and intercept error
reverse_curr_fits = []

for wavelength in wavelengths:
    region = reverse_stats[reverse_stats["Wavelength_nm"] == wavelength]

    V = region["Voltage_V"].to_numpy()
    I = region["mean_I_nA"].to_numpy()

    popt, pcov = curve_fit(linear, V, I)

    m, b = popt

    reverse_curr_fits.append({
        "Wavelength_nm": wavelength,
        "slope_nA_per_V": m,
        "intercept_nA": b,
    })

reverse_curr_fits = pd.DataFrame(reverse_curr_fits)
# print(reverse_curr_fits)

# now visually inspect these fits
plt.style.use("dark_background")
colors = sns.color_palette("husl", 5)

for wavelength, color in zip(wavelengths, colors):
    region = reverse_stats[reverse_stats["Wavelength_nm"] == wavelength]
    fit = reverse_curr_fits[reverse_curr_fits["Wavelength_nm"] == wavelength].iloc[0]

    m = fit["slope_nA_per_V"]
    b = fit["intercept_nA"]

    V_fit = np.linspace(
        region["Voltage_V"].min(),
        region["Voltage_V"].max(),
        100
    )

    plt.errorbar(
        region["Voltage_V"],
        region["mean_I_nA"],
        yerr=region["std_I_nA"],
        color="lightsteelblue",
        ecolor="white",
        fmt="o",
        capsize=3,
    )

    plt.plot(
        V_fit,
        linear(V_fit, m, b),
        color=color,
        label=f"{wavelength} nm"
    )

    plt.xlabel("Retarding Voltage (V)")
    plt.ylabel("Current (nA)")
    plt.title("Reverse Current Fits")
    plt.legend()
    plt.tight_layout()

plt.savefig("/home/rck67/jlab/photoelectric/figures/reverse_current_fits.png", dpi=300)

# if linear fits look reasonable, then add this to our data and proceed with normalization
data = data.merge(
    reverse_curr_fits,
    on="Wavelength_nm",
    how="left",
    validate="many_to_one" # multiple rows will be getting the same reverse current fit
)