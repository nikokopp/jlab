
import pandas as pd
from pathlib import Path

# Directory containing all seven joint-fit results
result_dir = Path("data/processed/joint_energy_fit")

# Find each magnetic field's energy summary
files = sorted(result_dir.glob("B*_*/energy_summary.csv"))

if not files:
    raise FileNotFoundError("No energy summary files found.")

# Combine all results
data = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

# Check that there are seven unique magnetic fields
if len(data) != 7 or data["B_G"].duplicated().any():
    raise ValueError(
        "Expected exactly seven unique magnetic fields. "
        f"Found {len(data)} rows:\n"
        f"{data['B_G'].tolist()}"
    )

# Select only the quantities needed for analysis
columns = [
    "B_G",
    "date",

    # Kinetic energy (deposited)
    "K_deposited_keV",
    "K_stat_err_keV",
    "K_cal_err_keV",
    "K_combined_err_keV",

    # Velocity
    "beta",
    "beta_err_measurement_approx",
    "beta_err_gap_common_sys",

    # Lorentz factor
    "gamma",
    "gamma_err_measurement_approx",
    "gamma_err_gap_common_sys"
]

data = data[columns].sort_values("B_G").reset_index(drop=True)

# Save one combined CSV
output = Path("data/processed/energy_velocity_results.csv")
output.parent.mkdir(parents=True, exist_ok=True)

data.to_csv(output, index=False)

print(data.to_string(index=False))
print(f"\nSaved results to {output}")
