"""
The kinetic-energy plots contain THEORY ONLY until measured MCA kinetic energies are entered.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c, e, m_e


root = Path(__file__).resolve().parents[1]
data_dir = root / 'data/processed/relativistic_relations'
figure_dir = root / 'figures/relativistic_relations'
figure_dir.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(data_dir / 'velocity_momentum_results.csv')
measured_file = data_dir / 'measured_kinetic_energies.csv'

# Optional independent MCA kinetic energies, one measurement per nominal B.
if measured_file.exists():
    measured = pd.read_csv(measured_file)
    required = {'B_nominal_G', 'K_keV', 'K_err_keV'}
    if not required.issubset(measured.columns):
        raise ValueError(f'{measured_file} must contain {sorted(required)}')
    if measured['B_nominal_G'].duplicated().any():
        raise ValueError('Kinetic-energy file has duplicate magnetic-field values.')
    results = results.merge(measured[list(required)], on='B_nominal_G', how='left', validate='one_to_one')
else:
    results['K_keV'] = np.nan
    results['K_err_keV'] = np.nan

has_K = results['K_keV'].notna()
if has_K.any() and not results.loc[has_K, 'K_err_keV'].notna().all():
    raise ValueError('Each measured K requires its uncertainty K_err_keV.')

mc2_keV = m_e * c**2 / (1e3 * e)


def finish(ax, filename, theory_only=False):
    if theory_only:
        ax.text(0.98, 0.04, 'Theory only, measurements not yet entered',
                ha='right', transform=ax.transAxes, fontsize=8, color='0.35')
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / filename, dpi=220)
    plt.close(ax.figure)
    print('Saved', figure_dir / filename)


def measured_points(ax, x_col, x_err_col):
    if not has_K.any():
        return
    ax.errorbar(results.loc[has_K, x_col], results.loc[has_K, 'K_keV'],
                xerr=results.loc[has_K, x_err_col],
                yerr=results.loc[has_K, 'K_err_keV'],
                fmt='o', color='black', capsize=3,
                label='Measured MCA kinetic energy')


# 1. Experimental momentum vs. selector-measured velocity.
beta_grid = np.linspace(0, min(0.95, max(0.90, results['beta'].max() + 0.06)), 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(beta_grid, mc2_keV * beta_grid / np.sqrt(1 - beta_grid**2),
        color='midnightblue', label='Relativistic prediction')
ax.plot(beta_grid, mc2_keV * beta_grid, color='cornflowerblue', ls='--',
        label='Classical prediction')
ax.errorbar(results['beta'], results['pc_keV'],
            xerr=results['beta_err'], yerr=results['pc_err_keV'],
            fmt='o', color='black', capsize=3, label='Experimental')
ax.set(xlabel=r'Electron velocity $\beta=v/c$',
       ylabel=r'Momentum $p$ (keV/$c$)', title='Momentum vs. velocity')
finish(ax, 'momentum_vs_beta.png')


# 2. Kinetic energy vs. momentum (pc in keV).
pc_grid = np.linspace(0, 1.13 * results['pc_keV'].max(), 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pc_grid, np.sqrt(pc_grid**2 + mc2_keV**2) - mc2_keV,
        color='midnightblue', label='Relativistic prediction')
ax.plot(pc_grid, pc_grid**2 / (2 * mc2_keV),
        color='cornflowerblue', ls='--', label='Classical prediction')
measured_points(ax, 'pc_keV', 'pc_err_keV')
ax.set(xlabel=r'Momentum $p$ (keV/$c$)',
       ylabel='Kinetic energy (keV)', title='Kinetic energy vs. momentum')
finish(ax, 'energy_vs_momentum.png', theory_only=not has_K.any())


# 3. Kinetic energy vs. beta.
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(beta_grid, mc2_keV * (1 / np.sqrt(1 - beta_grid**2) - 1),
        color='midnightblue', label='Relativistic prediction')
ax.plot(beta_grid, 0.5 * mc2_keV * beta_grid**2,
        color='cornflowerblue', ls='--', label='Classical prediction')
measured_points(ax, 'beta', 'beta_err')
ax.set(xlabel=r'Electron velocity $\beta=v/c$',
       ylabel='Kinetic energy (keV)', title='Kinetic energy vs. velocity')
finish(ax, 'energy_vs_beta.png', theory_only=not has_K.any())


# 4. Kinetic energy vs. gamma; classical expression is 1/2 m v^2,
# with v/c = sqrt(1 - 1/gamma^2) for comparison on the same x axis.
gamma_grid = np.linspace(1, 1.1 * results['gamma'].max(), 500)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(gamma_grid, mc2_keV * (gamma_grid - 1),
        color='midnightblue', label='Relativistic prediction')
ax.plot(gamma_grid, 0.5 * mc2_keV * (1 - 1 / gamma_grid**2),
        color='cornflowerblue', ls='--', label='Classical prediction')
measured_points(ax, 'gamma', 'gamma_err')
ax.set(xlabel=r'Lorentz factor $\gamma$',
       ylabel='Kinetic energy (keV)', title=r'Kinetic energy vs. $\gamma$')
finish(ax, 'energy_vs_gamma.png', theory_only=not has_K.any())

print(f'\nExperimental K values available: {has_K.sum()}/{len(results)}')
if not has_K.any():
    print('The three kinetic-energy figures show theory curves only.')
print('Measurements share systematic field/geometry uncertainties; error bars are approximate.')
