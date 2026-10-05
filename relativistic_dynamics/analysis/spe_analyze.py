import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from pathlib import Path

def read_spe(filename):
    path = Path(filename)
    lines = path.read_text().splitlines()

    i_time = lines.index("$MEAS_TIM:")
    live_time_s, real_time_s = map(float, lines[i_time + 1].split())

    i_data = lines.index("$DATA:")
    first_channel, last_channel = map(int, lines[i_data + 1].split())

    n_channels = last_channel - first_channel + 1

    counts = np.array(
        [int(line.strip()) for line in lines[i_data + 2 : i_data + 2 + n_channels]],
        dtype=int,
    )

    channels = np.arange(first_channel, last_channel + 1)

    if "$MCA_CAL:" in lines:
        i_cal = lines.index("$MCA_CAL:")
        n_coeffs = int(lines[i_cal + 1].strip())

        calibration_line = lines[i_cal + 2].split()
        coeffs = np.array(
            [float(x) for x in calibration_line[:n_coeffs]], dtype=float
        )
        energy_unit = calibration_line[n_coeffs]
    else:
        i_cal = lines.index("$ENER_FIT:")
        offset, slope = map(float, lines[i_cal + 1].split())
        coeffs = np.array([offset, slope])
        energy_unit = "keV"

    energy = np.zeros_like(channels, dtype=float)
    for power, coeff in enumerate(coeffs):
        energy += coeff * channels**power

    return {
        "path": path,
        "channels": channels,
        "counts": counts,
        "energy_keV": energy,
        "live_time_s": live_time_s,
        "real_time_s": real_time_s,
        "calibration": coeffs,
        "energy_unit": energy_unit,
    }

def parse_run_name(filename):
    """
    Extract B, selector voltage, and run number from filenames.
    """
    stem = Path(filename).stem

    pattern = (
        r"B(?P<B>\d+)G_"
        r"Vsel(?P<v_integer>\d+)p(?P<v_decimal>\d+)kV_"
        r"r(?P<run>\d+)"
    )

    match = re.search(pattern, stem)

    if match is None:
        return None

    B_G = float(match.group("B"))
    Vsel_kV = float(
        f"{match.group('v_integer')}.{match.group('v_decimal')}"
    )
    run = int(match.group("run"))

    return {
        "B_G": B_G,
        "Vsel_kV": Vsel_kV,
        "run": run,
    }

def roi_stats(spectrum, channel_min, channel_max):
    """
    Calculate useful statistics inside a chosen channel region
    """
    channels = spectrum["channels"]
    counts = spectrum["counts"]
    energy = spectrum["energy_keV"]
    live_time = spectrum["live_time_s"]

    mask = (channels >= channel_min) & (channels <= channel_max)

    roi_channels = channels[mask]
    roi_counts = counts[mask]
    roi_energy = energy[mask]

    total_counts = int(np.sum(roi_counts))
    count_rate = total_counts / live_time
    count_rate_error = np.sqrt(total_counts) / live_time

    if total_counts == 0:
        median_channel = np.nan
        median_energy_keV = np.nan
    else:
        cumulative_counts = np.cumsum(roi_counts)
        median_index = np.searchsorted(cumulative_counts, total_counts / 2)

        median_channel = roi_channels[median_index]
        median_energy_keV = roi_energy[median_index]

    return {
        "total_counts": total_counts,
        "count_rate_Hz": count_rate,
        "count_rate_error_Hz": count_rate_error,
        "median_channel": median_channel,
        "median_energy_keV": median_energy_keV,
    }

def plot_spectrum(spectrum):
    plt.figure()

    plt.step(
        spectrum["energy_keV"],
        spectrum["counts"],
        where="mid",
    )

    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.title(spectrum["path"].name)
    plt.tight_layout()
    plt.show()

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
    folder = "/home/rck67/jlab/relativistic_dynamics/data/raw/velocity_sweep/2026-10-05/"

    plot_velocity_sweep(
        folder=folder,
        B_target=90,
        channel_min=0,
        channel_max=2048,
    )