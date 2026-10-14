import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from spe_analyzer import read_spe, parse_run_name, roi_stats

data_root = Path("/home/rck67/jlab/relativistic_dynamics/data/raw/calibration")
data_file = Path(data_root / "2026-10-14/20261014_background_calibration.Spe")
figure_dir = Path("/home/rck67/jlab/relativistic_dynamics/figures/calibration")
output_dir = Path("/home/rck67/jlab/relativistic_dynamics/data/processed/calibration_results")
figure_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

spectrum = read_spe(data_file)

channels = spectrum["channels"]
counts = spectrum["counts"]
live_time = spectrum["live_time_s"]
counts_err = np.sqrt(counts)

print(f"Number of channels: {len(channels)}")
print(f"Total counts: {np.sum(counts)}")
print(f"Live time: {live_time:.1f} s")

fig, ax1 = plt.subplots(figsize=(10, 7))

ax1.step(
        channels, 
        counts,
        color="midnightblue")

ax1.set(
    xlabel="Channel",
    ylabel="Counts per channel",
    title=f"Ba-133 background calibration ({live_time:.2f} s)",
    xlim=(0, 2048),
    yscale="log"
)

fig.tight_layout()
fig.savefig(figure_dir / "ba133_background_calibration_raw_spectrum.png")
plt.show()
