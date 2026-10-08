import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

from pathlib import Path
from spe_analyze import read_spe, parse_run_name, roi_stats, plot_spectrum

def plot_velocity_sweep(folder, B_target, channel_min, channel_max):
    folder = Path(folder)

    voltages = []
    rates = []
    rate_errors = []

    for filename in folder.glob("*.Spe"):

        run_info = parse_run_name(filename)

        # ignore files whose names don't match our expected pattern
        if run_info is None:
            continue

        # only use files taken at the magnetic field we want
        if run_info["B_G"] != B_target:
            continue

        spectrum = read_spe(filename)

        stats = roi_stats(
            spectrum,
            channel_min,
            channel_max,
        )

        voltages.append(run_info["Vsel_kV"])
        rates.append(stats["count_rate_Hz"])
        rate_errors.append(stats["count_rate_error_Hz"])

        print(
            f"Vsel = {run_info['Vsel_kV']} kV | "
            f"counts = {stats['total_counts']} | "
            f"rate = {stats['count_rate_Hz']} +/- "
            f"{stats['count_rate_error_Hz']} counts/s"
        )

    # convert to numpy arrays
    voltages = np.array(voltages)
    rates = np.array(rates)
    rate_errors = np.array(rate_errors)

    # sort everything by voltage
    order = np.argsort(voltages)

    voltages = voltages[order]
    rates = rates[order]
    rate_errors = rate_errors[order]

    # plot
    plt.figure()

    plt.errorbar(
        voltages,
        rates,
        yerr=rate_errors,
        fmt="o-",
        capsize=3,
    )

    plt.xlabel("Velocity selector voltage (kV)")
    plt.ylabel("Count rate (counts/s)")
    plt.title(f"Voltage sweep at B = {B_target} G")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = "/home/rck67/jlab/relativistic_dynamics/data/raw/velocity_sweep/2026-10-07/"

    plot_velocity_sweep(
        folder=folder,
        B_target=100,
        channel_min=0,
        channel_max=2048,
    )

