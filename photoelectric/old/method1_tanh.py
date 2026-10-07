import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit, brentq

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
stats_file = Path("/home/rck67/jlab/photoelectric/data/current_statistics.csv")

output_dir = Path("/home/rck67/jlab/photoelectric/")
figure_dir = output_dir / "figures/method1"

output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

wavelengths = [546.1, 435.7, 365.0]

reverse_min_V = {
    546.1: 6.0,
    435.7: 6.0,
    365.0: 7.0,
}

data = pd.read_csv(data_file)
stats = pd.read_csv(stats_file)

# average any duplicate measurements (consistent with current_statistics.py)
per_run = (
    data
    .groupby(
        ["Wavelength_nm", "Voltage_V", "Run"],
        as_index=False
    )
    .agg(
        Current_nA=("Current_nA", "mean")
    )
)

# we will find the intersection between fitted tanh and fitted reverse current
# define models
def linear(V, m, b):
    return m*V + b

def tanh_model(V, A, V0, w, I_low):
    return(I_low + 0.5*A*(1 - np.tanh((V-V0)/w)))

# load reverse current fits
reverse_curr_fits = pd.read_csv("/home/rck67/jlab/photoelectric/data/reverse_current_fits.csv")

# helper to find the zero crossing point
def find_zero_crossing(V, I):
    """
    Find first positive to negative crossing of measured current using linear interpolation.
    """
    for i in range(len(V) - 1):
        # point exactly at zero
        if np.isclose(I[i], 0.0):
            return V[i]
        
        # positive to negative crossing, use linear interpolation
        if I[i] > 0 and I[i + 1] < 0:
            V_zero = (V[i] - I[i] * (V[i + 1] - V[i]) / (I[i + 1] - I[i]))
            return V_zero

    return np.nan

# helper to find tanh and reverse current intersection
def find_cutoff(V_zero, V_max, tanh_params, reverse_params):
    A, V0, w, I_low = tanh_params
    m, b = reverse_params

    def difference(V):
        return (tanh_model(V, A, V0, w, I_low) - linear(V, m, b))

    # do not accidentally return exactly the measured zero crossing point
    V_start = V_zero + 1e-6

    if V_start >= V_max:
        return np.nan

    # search for sign change between the two fitted models
    V_grid = np.linspace(V_start, V_max, 5000)

    diff = difference(V_grid)

    crossing_indices = np.where(diff[:-1] * diff[1:] < 0)[0]

    if len(crossing_indices) == 0:
        return np.nan

    # first fitted model intersection after I = 0
    i = crossing_indices[0]
    Vc = brentq(difference, V_grid[i], V_grid[i + 1])

    return Vc

fit_results = []

plt.style.use("dark_background")

# fit tanh to each run
for wavelength in wavelengths:
    wavelength_data = per_run[per_run["Wavelength_nm"] == wavelength]

    # get reverse current fit for this wavelength
    reverse_fit = reverse_curr_fits[reverse_curr_fits["Wavelength_nm"] == wavelength].iloc[0]

    m = reverse_fit["slope_nA_per_V"]
    b = reverse_fit["intercept_nA"]

    runs = sorted(wavelength_data["Run"].unique())

    for run in runs:

        run_data = wavelength_data[wavelength_data["Run"] == run].sort_values("Voltage_V")

        V = run_data["Voltage_V"].to_numpy()
        I = run_data["Current_nA"].to_numpy()

        # initial guesses for tanh
        I0 = I[0]
        I_low0 = I[-1]
        A0 = I0 - I_low0
        I_mid0 = I_low0 + A0 / 2

        V0_guess = V[np.argmin(np.abs(I - I_mid0))]
        w_guess = 1.0

        # fit tanh

        popt, pcov = curve_fit(
            tanh_model,
            V,
            I,
            p0=[A0, V0_guess, w_guess, I_low0],
            bounds=([0.0, 0.0, 0.01, -np.inf], [np.inf, V.max(), 10.0, np.inf]),
            maxfev=50000
        )

        A, V0, w, I_low = popt
        A_err, V0_err, w_err, I_low_err = np.sqrt(np.diag(pcov))

        # find measured zero crossing point
        V_zero = find_zero_crossing(V, I)

        if np.isnan(V_zero):
            print(
                f"WARNING: no measured zero crossing for "
                f"{wavelength} nm, run {run}"
            )

            Vc = np.nan

        else:
            Vc = find_cutoff(V_zero, V.max(), popt, (m, b))

            if np.isnan(Vc):
                print(
                    f"WARNING: no tanh/reverse current intersection after I=0 for"
                    f"{wavelength} nm, run {run}"
                )


        # fit error and RMSE
        I_fit = tanh_model(V, *popt)
        residuals = I - I_fit
        rmse = np.sqrt(np.mean(residuals**2))

        # save results

        fit_results.append({
            "Wavelength_nm": wavelength,
            "Run": run,

            "A_nA": A,
            "A_err_nA": A_err,

            "V0_V": V0,
            "V0_err_V": V0_err,

            "w_V": w,
            "w_err_V": w_err,

            "I_low_nA": I_low,
            "I_low_err_nA": I_low_err,

            "reverse_slope_nA_per_V": m,
            "reverse_intercept_nA": b,

            "V_zero_V": V_zero,
            "Vc_V": Vc,

            "RMSE_nA": rmse,
        })


        # plot fit

        V_plot = np.linspace(0.0, V.max(), 1000)

        plt.figure(figsize=(8, 5))

        # measured data
        plt.scatter(
            V,
            I,
            color="cornflowerblue",
            s=25,
            label="Measured current"
        )

        # tanh fit
        plt.plot(
            V_plot,
            tanh_model(V_plot, *popt),
            color="lightsteelblue",
            label="Tanh fit"
        )

        # reverse current fit
        # fitted region is solid, backward extrapolation is dashed
        V_rev_solid = np.linspace(reverse_min_V[wavelength], V.max(), 200)

        plt.plot(
            V_rev_solid,
            linear(
                V_rev_solid,
                m,
                b
            ),
            color="white",
            label="Reverse current linear fit"
        )

        if (
            not np.isnan(V_zero)
            and V_zero < reverse_min_V[wavelength]
        ):

            V_rev_extrap = np.linspace(
                V_zero,
                reverse_min_V[wavelength],
                200
            )

            plt.plot(
                V_rev_extrap,
                linear(
                    V_rev_extrap,
                    m,
                    b
                ),
                color="white",
                linestyle="--",
                label="Reverse current extrapolation"
            )

        # measured zero crossing
        if not np.isnan(V_zero):
            plt.axvline(
                V_zero,
                color="gray",
                linestyle=":",
                linewidth=1,
                label=fr"$I=0$: {V_zero:.3f} V"
            )

        # method 1 final cutoff
        if not np.isnan(Vc):
            plt.axvline(
                Vc,
                color="plum",
                linestyle="--",
                linewidth=1.5,
                label=fr"$V_c$: {Vc:.3f} V"
            )

            # intersection point
            plt.scatter(
                Vc,
                linear(Vc, m, b),
                color="plum",
                s=45,
                zorder=5
            )


        plt.axhline(
            0,
            color="gray",
            linewidth=0.8
        )


        plt.xlabel("Retarding Voltage (V)")
        plt.ylabel("Current (nA)")
        plt.title(f"Method 1: {wavelength} nm, Run {run}")
        plt.legend(fontsize=6)
        plt.tight_layout()

        plt.savefig(
            figure_dir
            / f"individual_runs/tanh_{wavelength}nm_run{run}.png",
            dpi=300
        )
        print(f"Saved plot to figures/method1/individual_runs/tanh_{wavelength}nm_run{run}.png")
        plt.close()

fit_results = pd.DataFrame(fit_results)
fit_results = fit_results.sort_values(["Wavelength_nm", "Run"]).reset_index(drop=True)

# save our fits and cutoff voltages, yay!!
fit_results.to_csv(output_dir / "data/method1_tanh_cutoffs.csv", index=False)
print("Saved method 1 results to data/method1_tanh_cutoffs.csv")

cutoff_stats = (
    fit_results
    .groupby("Wavelength_nm")["Vc_V"]
    .agg(
        mean_Vc_V="mean",
        std_Vc_V="std",
        N="count"
    )
    .reset_index()
)

cutoff_stats["SEM_Vc_V"] = cutoff_stats["std_Vc_V"] / np.sqrt(cutoff_stats["N"])

cutoff_stats.to_csv("/home/rck67/jlab/photoelectric/data/method1_tanh_cutoff_stats.csv", index=False)
print("Saved cutoff voltage statistics to /data/method1_tanh_cutoff_stats.csv")