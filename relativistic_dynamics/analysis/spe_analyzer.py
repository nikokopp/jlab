import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path

def read_spe(filename):
    """Read .spe file, including header and footer data."""
    path = Path(filename)
    lines = path.read_text().splitlines()

    i_time = lines.index("$MEAS_TIM:")
    live_time_s, real_time_s = map(float, lines[i_time + 1].split())

    i_data = lines.index("$DATA:")
    first_channel, last_channel = map(int, lines[i_data + 1].split())

    n_channels = last_channel - first_channel + 1
    channels = np.arange(first_channel, last_channel + 1)
    
    i_counts = i_data + 2
    counts = np.array([int(line.strip()) for line in lines[i_counts : i_counts + n_channels]], dtype=int)

    return {
        "path": path,
        "channels": channels,
        "counts": counts,
        "live_time_s": live_time_s,
        "real_time_s": real_time_s
    }

def parse_run_name(filename):
    """
    Extract B, velocity selector voltage, and run number from filenames.
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
    Calculate useful statistics inside a chosen channel region (region of interest, aka roi).
    """
    channels = spectrum["channels"]
    counts = spectrum["counts"]
    live_time = spectrum["live_time_s"]

    mask = (channels >= channel_min) & (channels <= channel_max)

    roi_channels = channels[mask]
    roi_counts = counts[mask]

    total_counts = int(np.sum(roi_counts))
    count_rate = total_counts / live_time
    count_rate_error = np.sqrt(total_counts) / live_time

    if total_counts == 0:
        median_channel = np.nan
    else:
        cumulative_counts = np.cumsum(roi_counts)
        median_index = np.searchsorted(cumulative_counts, total_counts / 2)

        median_channel = roi_channels[median_index]

    return {
        "total_counts": total_counts,
        "count_rate_Hz": count_rate,
        "count_rate_error_Hz": count_rate_error,
        "median_channel": median_channel
    }

def plot_spectrum(spectrum):
    plt.figure()

    plt.step(
        spectrum["channels"],
        spectrum["counts"],
        where="mid",
    )

    plt.xlabel("Channel")
    plt.ylabel("Counts")
    plt.title(spectrum["path"].name)
    plt.tight_layout()
    plt.show()