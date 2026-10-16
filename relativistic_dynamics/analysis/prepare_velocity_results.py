"""Recover measured velocity and momentum results from the six fitted selector voltages.

Run from the relativistic_dynamics project root:
    python analysis/prepare_velocity_results.py

The voltages and magnetic-field corrections are transcribed from elliott_analysis.ipynb.
Units are SI except where names explicitly say G, kV, or keV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import c, e, m_e


root = Path(__file__).resolve().parents[1]
output_dir = root / 'data/processed/relativistic_relations'

# Fitted symmetric-trapezoid velocity-selector results (partner's notebook, cell 8).
B_nominal_G = np.array([80, 89.4, 91.9, 100, 110, 120], dtype=float)
V0_kV = np.array([2.6688, 3.2185, 3.3592, 3.7834, 4.2964, 4.8713])
V0_err_kV = np.array([0.0121, 0.0145, 0.0086, 0.0112, 0.0142, 0.0115])

# Field corrections recorded in the notebook. Each pair is (+B, -B) calibration.
mag_G = 99.69
calibration_pairs = np.array([
    [104.0, -102.2],
    [106.0, -103.6],
    [105.0, -105.4],
    [104.0, -102.2],
    [105.1, -106.2],
    [107.3, -107.4],
])
zero_G = np.mean(calibration_pairs, axis=1)
scale = np.mean(np.abs(calibration_pairs), axis=1) / mag_G
B_corrected_G = (B_nominal_G - zero_G) / scale

# from Elliott's analysis
# The 21.374856 slope comes field vs. current calibration
field_current_slope_G_per_A = 21.374856034068436
sigma_B_nominal_G = field_current_slope_G_per_A * 0.2
sigma_zero_G = np.sqrt(2 * 0.02**2) / 2
sigma_scale = np.sqrt((2 * 0.02**2) / (2 * mag_G)**2 + (0.01 * scale)**2)
B_err_G = np.sqrt(sigma_B_nominal_G**2 + sigma_zero_G**2
                  + B_corrected_G**2 * sigma_scale**2) / scale

# Apparatus dimensions
d_m = 0.00180
d_err_m = 0.00003
rho_m = 0.203
rho_err_m = 0.002

B_T = B_corrected_G * 1e-4
V0_V = V0_kV * 1000
beta = V0_V / (d_m * B_T * c)
beta_err = beta * np.sqrt((V0_err_kV / V0_kV)**2
                          + (d_err_m / d_m)**2
                          + (B_err_G / B_corrected_G)**2)
if np.any(beta >= 1):
    raise ValueError('An inferred speed exceeds c. Check inputs and units.')

gamma = 1 / np.sqrt(1 - beta**2)
gamma_err = beta * gamma**3 * beta_err

p_SI = e * B_T * rho_m
# Numerical value of pc in keV, equivalent to p in keV/c
pc_keV = p_SI * c / (1e3 * e)
pc_err_keV = pc_keV * np.sqrt((B_err_G / B_corrected_G)**2
                                    + (rho_err_m / rho_m)**2)

results = pd.DataFrame({
    'B_nominal_G': B_nominal_G,
    'B_corrected_G': B_corrected_G,
    'B_err_G': B_err_G,
    'V0_kV': V0_kV,
    'V0_err_kV': V0_err_kV,
    'beta': beta,
    'beta_err': beta_err,
    'gamma': gamma,
    'gamma_err': gamma_err,
    'pc_keV': pc_keV,
    'pc_err_keV': pc_err_keV,
    'p_kg_m_per_s': p_SI,
})

output_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(output_dir / 'velocity_momentum_results.csv', index=False)
print(results[['B_nominal_G', 'B_corrected_G', 'V0_kV',
               'beta', 'gamma', 'pc_keV']].to_string(index=False, float_format='%.5f'))
print(f'\nSaved {output_dir / "velocity_momentum_results.csv"}')
print('Uncertainties are preliminary and follow the notebook prescription.')
