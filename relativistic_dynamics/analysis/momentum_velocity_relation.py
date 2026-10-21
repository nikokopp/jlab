from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.constants import c, elementary_charge, electron_mass

data_file = Path("/home/rck67/jlab/relativistic_dynamics/data/processed/trapezoid_fits/velocity_sweep_results.csv")
data = pd.read_csv(data_file)

B = data['B_corrected_G'].to_numpy(dtype=float)
B_err = data['Berr_corrected_G'].to_numpy(dtype=float)
V0_kV = data['V_opt_kV'].to_numpy(dtype=float)
V0_stat_kV = data['V_opt_stat_err_kV'].to_numpy(dtype=float)

# convert from E/B to beta and gamma*beta
d_cm = 0.180
rho_m = 0.406 / 2
volts_per_statvolt = 299.792458
E = V0_kV * 1000 / volts_per_statvolt / d_cm
E_err = V0_stat_kV * 1000 / volts_per_statvolt / d_cm
beta = E / B
gamma = 1 / np.sqrt(1-beta**2)
gamma_beta = beta * gamma

beta_err_V = E_err / B
gamma_beta_err_V = gamma**3 * beta_err_V

# Convert B (gauss) to momentum (keV/c) for plotting only.
# In SI: p = e B rho; B[T] = 1e-4 * B[G].
# The electron charge cancels when converting kg m/s to keV/c.
p_factor = c * rho_m * 1e-7  # (keV/c) / G
p = p_factor * B
p_err_B = p_factor * B_err

# The orbit radius is 0.203 +/- 0.002 m. Its error is common to all
# measurements: include it on the displayed error bars, not independently
# in the B-based fit (which below remains unchanged).
rho_err_m = 0.004 / 2
p_err_rho = p * (rho_err_m / rho_m)
p_err_total = np.hypot(p_err_B, p_err_rho)

# B in both the y axis and x axis as (E/B), so treat correlation based on model
def f_and_derivative(beta, model):
    if model == 'nonrel':
        return beta, np.ones_like(beta)
    if model == 'rel':
        return beta / np.sqrt(1-beta**2), (1-beta**2)**(-1.5)
    raise ValueError(model)

# function for residuals
def standardized_residuals(a, model):
    f, f_prime = f_and_derivative(beta, model)
    delta = B-a*f

    # dr/dB = 1 + a*f'(beta)*beta/B
    # dr/dV = -a*f'(beta)*E_err/B (when multiplied by sigma_V)
    delta_err = np.hypot(
        (1+a*f_prime*beta/B)*B_err,
        a*f_prime*beta_err_V,
    )
    return delta / delta_err

def fit_slope(model):
    result = least_squares(
        lambda pars: standardized_residuals(pars[0], model),
        x0=[85.0], bounds=(0.0, np.inf)
    )
    a = result.x[0]
    chi2 = float(np.sum(result.fun**2))
    dof = len(B)-1

    a_err = float(np.sqrt(np.linalg.inv(result.jac.T @ result.jac)[0, 0]))
    return a, a_err, chi2, dof

nonrel_fit = fit_slope('nonrel')
rel_fit = fit_slope('rel')

# Each model gets its own best-fit slope and inferred e/m.
a_nonrel, a_nonrel_err = nonrel_fit[:2]
a_rel, a_rel_err = rel_fit[:2]

def slope_to_em(a, a_err):
    # Gaussian B in gauss, rho in metres; convert e/m to SI (C/kg).
    em_C_kg = 1e4 * c / (rho_m * a)
    em_err_C_kg = em_C_kg * a_err / abs(a)
    return em_C_kg, em_err_C_kg

em_nonrel, em_nonrel_err = slope_to_em(a_nonrel, a_nonrel_err)
em_rel, em_rel_err = slope_to_em(a_rel, a_rel_err)

for model, fit, em, em_err in [
    ('Nonrelativistic', nonrel_fit, em_nonrel, em_nonrel_err),
    ('Relativistic', rel_fit, em_rel, em_rel_err),
]:
    a_fit, a_fit_err, chi2, dof = fit
    print(f'{model} fit: a = {a_fit:.3f} +/- {a_fit_err:.3f} G; '
          f'chi2/dof = {chi2:.2f}/{dof} = {chi2/dof:.2f}')
    print(f'  e/m = ({em:.5e} +/- {em_err:.3e}) C/kg (fit uncertainty only)')

fit_results = pd.DataFrame([
    {
        "model": "Nonrelativistic",
        "slope": nonrel_fit[0],
        "slope_err": nonrel_fit[1],
        "e_over_m_C_kg": em_nonrel,
        "e_over_m_err_C_kg": em_nonrel_err,
        "chi2": nonrel_fit[2],
        "dof": nonrel_fit[3],
        "reduced_chi2": nonrel_fit[2] / nonrel_fit[3]
    },
    {
        "model": "Relativistic",
        "slope": rel_fit[0],
        "slope_err": rel_fit[1],
        "e_over_m_C_kg": em_rel,
        "e_over_m_err_C_kg": em_rel_err,
        "chi2": rel_fit[2],
        "dof": rel_fit[3],
        "reduced_chi2": rel_fit[2] / rel_fit[3]
    }
])

Path('/home/rck67/jlab/relativistic_dynamics/data/processed/relativistic_relations').mkdir(parents=True, exist_ok=True)
fit_results.to_csv(
    "/home/rck67/jlab/relativistic_dynamics/data/processed/relativistic_relations/momentum_velocity_fit_results.csv",
    index=False
)

colors = {
    "data": "midnightblue",
    "error": "midnightblue",
    "rel": "mediumpurple",
    "nonrel": "steelblue"
}

fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")

# show axes to zero to show agreement at low velocities
low_beta = np.linspace(0.0, min(0.90, max(beta)+0.055), 500)
low_gammabeta = np.linspace(0.0, max(gamma_beta)+0.2, 500)

# plot momentum vs beta (fit remains in B)
# measurements
axs[0].errorbar(
    beta, p, 
    xerr=beta_err_V, 
    yerr=p_err_total, 
    fmt='o', 
    ms=4,
    color=colors['data'], 
    ecolor=colors['error'], 
    capsize=3,
    elinewidth=1.2, 
    label='Measurements')

# plot relativistic fit
axs[0].plot(low_beta, p_factor*a_rel*low_beta/np.sqrt(1-low_beta**2),
            color=colors['rel'], lw=2,
            label='Relativistic fit')

# plot nonrelativistic fit
axs[0].plot(low_beta, p_factor*a_nonrel*low_beta, ls='--', lw=2,
            color=colors['nonrel'], label='Nonrelativistic fit')

axs[0].set_xlabel(r'$\beta = E/B = v/c$')
axs[0].set_ylabel(r'Momentum $p$ (keV/$c$)')
axs[0].set_xlim(left=0)
axs[0].set_title(r'Momentum vs. $\beta$')

# plot momentum vs gamma*beta (not gamma alone)
# plot measurements
axs[1].errorbar(gamma_beta, p, 
                xerr=gamma_beta_err_V, 
                yerr=p_err_total,
                fmt='o', 
                ms=4, 
                color=colors['data'], 
                ecolor=colors['error'],
                capsize=3, 
                elinewidth=1.2, 
                label='Measurements')

# relativistic fit
axs[1].plot(low_gammabeta, p_factor*a_rel*low_gammabeta, color=colors['rel'],
            lw=2, label='Relativistic fit')

# nonrelativistic fit
axs[1].plot(low_gammabeta, p_factor*a_nonrel*low_gammabeta/np.sqrt(1+low_gammabeta**2),
            ls='--', lw=2, color=colors['nonrel'],
            label='Nonrelativistic fit')

axs[1].set_xlabel(r'$\gamma\beta = \beta/\sqrt{1-\beta^2}$')
axs[1].set_ylabel(r'Momentum $p$ (keV/$c$)')
axs[1].set_xlim(left=0)
axs[1].set_title(r'Momentum vs. $\gamma\beta$')

for ax in axs:
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8.5)

    stats_text = (
        rf'Relativistic fit: $\chi^2/\mathrm{{dof}} = {rel_fit[2]:.3f}/{rel_fit[3]} = {rel_fit[2]/rel_fit[3]:.3f}$'
        "\n"
        rf'Nonrelativistic fit: $\chi^2/\mathrm{{dof}} = {nonrel_fit[2]:.3f}/{nonrel_fit[3]} = {nonrel_fit[2]/nonrel_fit[3]:.3f}$'
    )

    ax.text(
        0.98, 0.055,
        stats_text,
        ha='right',
        va='bottom',
        transform=ax.transAxes,
        fontsize=8.5,
        linespacing=1.5,
        bbox={
            'facecolor': 'white',
            'edgecolor': '0.7',
            'alpha': 0.9,
            'boxstyle': 'round,pad=0.4'
        }
    )
    
fig.tight_layout()

out_file = Path("/home/rck67/jlab/relativistic_dynamics/figures/relativistic_relations/momentum_velocity_independent_fits.png")

out_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_file, dpi=220, bbox_inches='tight')

print('Saved', out_file)

# Calculate e/m at each B, in C/kg
em_factor = 1e4 * c / rho_m

em_nonrel_points = em_factor * beta / B
em_rel_points = em_factor * gamma * beta / B

# Propagate measurement uncertainties
frac_V_err = V0_stat_kV / np.abs(V0_kV)
frac_B_err = B_err / np.abs(B)

em_nonrel_point_err = em_nonrel_points * np.sqrt(
    frac_V_err**2 + (2 * frac_B_err)**2
)

em_rel_point_err = em_rel_points * np.sqrt(
    (gamma**2 * frac_V_err)**2
    + ((1 + gamma**2) * frac_B_err)**2
)

# accepted value
em_reference = elementary_charge / electron_mass

# scale for nicer axis labels
scale = 1e11

fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')

# darker point colors
nonrel_point_color = "steelblue"
rel_point_color = "mediumpurple"

# lighter band colors
nonrel_band_color = "#cfe3ff"   # light blue
rel_band_color = "#e6d5ff"      # light purple

# shaded band: relativistic e/m ± std
ax.axhspan(
    (em_rel - em_rel_err) / scale,
    (em_rel + em_rel_err) / scale,
    color=rel_band_color,
    alpha=0.6,
    label=(
        fr'Relativistic e/m range: {em_rel / scale:.3f}' 
        fr'$\pm {em_rel_err / scale:.3f}_\mathrm{{stat}}$'
        r'$\times 10^{11}$ C/kg'
    )
)

# shaded band: nonrelativistic e/m ± std
ax.axhspan(
    (em_nonrel - em_nonrel_err) / scale,
    (em_nonrel + em_nonrel_err) / scale,
    color=nonrel_band_color,
    alpha=0.6,
    label=(
        fr'Nonrelativistic e/m range: {em_nonrel / scale:.3f}'
        fr'$\pm {em_nonrel_err / scale:.3f}_\mathrm{{stat}}$'
        r'$\times 10^{11}$ C/kg'
    )
)

# relativistic points
ax.errorbar(
    B,
    em_rel_points / scale,
    xerr=B_err,
    yerr=em_rel_point_err / scale,
    fmt='s',
    ms=5,
    color=rel_point_color,
    ecolor=rel_point_color,
    capsize=3,
    elinewidth=1.1,
    label='Relativistic points'
)

# nonrelativistic points
ax.errorbar(
    B,
    em_nonrel_points / scale,
    xerr=B_err,
    yerr=em_nonrel_point_err / scale,
    fmt='o',
    ms=5,
    color=nonrel_point_color,
    ecolor=nonrel_point_color,
    capsize=3,
    elinewidth=1.1,
    label='Nonrelativistic points'
)

# accepted value
ax.axhline(
    em_reference / scale,
    color='0.25',
    linestyle='--',
    linewidth=1.5,
    label=(
        fr'Accepted $e/m$: {em_reference / scale:.3f}'
        r'$\times 10^{11}$ C/kg'
    )
)

ax.set_xlabel(r'Magnetic field $B$ (G)')
ax.set_ylabel(r'$e/m$ ($10^{11}$ C/kg)')
ax.set_title(r'Calculated $e/m$ vs. magnetic field')
ax.grid(alpha=0.22)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2
)

out_file = Path(
    "/home/rck67/jlab/relativistic_dynamics/figures/relativistic_relations/em_vs_B_bands.png"
)
out_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_file, dpi=220, bbox_inches='tight')
print("Saved", out_file)

plt.show()