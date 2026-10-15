"""
Instructions on using this code for different calibrations:

First, inspect your calibration spectrum to determine what spectral features to enable in calibration_features.json:
    python analysis/mca_calibration_pipeline.py inspect --spe data.Spe --output results

Overnight:
    python analysis/mca_calibration_pipeline.py overnight --spe data.Spe \
        --config calibration_features.json --output results
Daily:
    python analysis/mca_calibration_pipeline.py daily --reference results/overnight_calibration.json \
        --files 'data/raw/calibration/*/*.Spe' --output results/daily

Code was developed with AI assistance.
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erfc, xlogy
from spe_analyzer import read_spe, parse_run_name

# emission intensities for 81.0 keV peak
E_81 = 80.997
E_79 = 79.614
R_79_81 = 2.62 / 34.06

# emission lines for 31.0 keV peak (multiple Cs K emission lines)
# emission line energies from NIST, intensities from XrayDB
CS_K_E = np.array([30.97313, 30.62540, 34.9873, 34.91968, 35.8126])
CS_K_REL = np.array([.53016, .28705, .09551, .04947, .03552]) / .53016
CS_K_ALPHA1 = float(CS_K_E[0])


def compton_edge_keV(gamma_energy_keV):
    """Maximum electron kinetic energy for 180-degree gamma scattering."""
    return gamma_energy_keV * (1 - 1 / (1 + 2 * gamma_energy_keV / 510.99895))

def model_components(x, params, kind, center, slope_guess, fixed_width=None, background_order=1, fixed_beta_scale=None):
    """Return the spectral feature and positive exponential background."""
    x = np.asarray(x, dtype=float)
    amp, mu = params[:2]

    idx = 2
    if fixed_width is None:
        width = params[idx]
        idx += 1
    else:
        width = fixed_width

    log_b0 = params[idx]
    bg_slope = params[idx + 1]
    bg_curve = params[idx + 2] if background_order == 2 else 0.0

    z = x - center
    background = np.exp(np.clip(log_b0 + bg_slope*z + bg_curve*z**2, -30, 30))

    if kind == 'gaussian':
        signal = amp * np.exp(-0.5 * ((x - mu) / width)**2)

    elif kind == 'cs_k_complex':
        beta_scale = params[-1] if fixed_beta_scale is None else fixed_beta_scale
        signal = np.zeros_like(x)

        for i, (energy, rel) in enumerate(zip(CS_K_E, CS_K_REL)):
            mu_i = mu + (energy - CS_K_ALPHA1) / slope_guess
            strength = rel * (beta_scale if i >= 2 else 1.0)
            signal += amp * strength * np.exp(-0.5 * ((x - mu_i) / width)**2)

    elif kind == 'doublet':
        delta = (E_81 - E_79) / slope_guess
        signal = amp * (np.exp(-0.5 * ((x - mu) / width)**2)
                        + R_79_81 * np.exp(-0.5 * ((x - (mu - delta)) / width)**2))

    elif kind == 'edge':
        signal = 0.5 * amp * erfc((x - mu) / (np.sqrt(2) * width))

    else:
        raise ValueError(f'Unknown feature kind: {kind}')

    return signal, background


def model(x, params, kind, center, slope_guess, fixed_width=None, background_order=1, fixed_beta_scale=None):
    signal, background = model_components(
        x, params, kind, center, slope_guess,
        fixed_width, background_order, fixed_beta_scale
    )
    return signal + background

def poisson_deviance_residuals(y, prediction):
    prediction = np.maximum(prediction, 1e-10)

    dev = 2 * (prediction - y + xlogy(y, y / prediction))

    return np.sign(y - prediction) * np.sqrt(np.maximum(dev, 0))

def fit_feature(spectrum, feature, slope_guess, *, fixed_width=None, fixed_beta_scale=None):
    lower, upper = map(float, feature['window'])

    xall = spectrum['channels']
    yall = spectrum['counts']

    mask = (xall >= lower) & (xall <= upper)
    x, y = xall[mask].astype(float), yall[mask].astype(float)

    if len(x) < 12 or np.sum(y) < 20:
        raise ValueError(f"Insufficient bins/counts for feature {feature['name']}")
    
    kind = feature['kind']
    order = int(feature.get('background_order', 1))

    if order not in (1, 2):
        raise ValueError('background_order must be 1 or 2')
    
    center = (lower + upper) / 2
    b0 = max(np.quantile(y, .20), .10)
    amp0 = max(np.max(y) - b0, 1.0)
    mu0 = float(feature.get('expected_channel', x[np.argmax(y)]))
    width0 = float(feature.get('width_guess', 4.0))

    init = [amp0, mu0]
    lo = [0., lower]
    hi = [max(1.e7, float(y.max()) * 100), upper]

    if fixed_width is None:
        init += [width0]
        lo += [.5]
        hi += [max(15., (upper - lower) / 2)]

    init += [np.log(b0), 0.]
    lo += [-15., -.5]
    hi += [max(20., np.log(max(y.max(),1.)) + 5), .5]

    if order == 2:
        init += [0.]
        lo += [-.025]
        hi += [.025]

    if kind == 'cs_k_complex' and fixed_beta_scale is None:
        init += [float(feature.get('beta_scale_guess', 1.0))]
        lo += [0.]
        hi += [3.0]

    init, lo, hi = np.asarray(init), np.asarray(lo), np.asarray(hi)
    init = np.maximum(lo + 1e-5, np.minimum(hi - 1e-5, init))

    def predict(p, xv=x):
        return model(xv, p, kind, center, slope_guess, fixed_width, order, fixed_beta_scale)

    outcome = least_squares(lambda p: poisson_deviance_residuals(y, predict(p)),
                            init, bounds=(lo, hi), max_nfev=10000,
                            x_scale='jac')
    
    if not outcome.success:
        raise RuntimeError(f"Fit failed for {feature['name']}: {outcome.message}")
    
    pred = predict(outcome.x)
    chi2 = float(np.sum(poisson_deviance_residuals(y, pred)**2))
    pearson_chi2 = float(np.sum((y - pred)**2 / np.maximum(pred, 1e-10)))
    dof = int(len(x) - len(outcome.x))

    try:
        covariance = np.linalg.inv(outcome.jac.T @ outcome.jac)
        mu_stat = float(np.sqrt(max(covariance[1, 1], 0)))

    except np.linalg.LinAlgError:
        mu_stat = float('nan')

    width = float(outcome.x[2]) if fixed_width is None else float(fixed_width)

    bound_hit = np.any(np.isclose(outcome.x, lo, atol=1e-3)) or np.any(
        np.isclose(outcome.x, hi, atol=1e-3))
    beta_scale = (float(outcome.x[-1]) if kind == 'cs_k_complex' and
                  fixed_beta_scale is None else fixed_beta_scale)
    
    return dict(name=feature['name'],
                display_name=feature.get('display_name', feature['name']),
                plot_label=feature.get('plot_label', feature.get('display_name', feature['name'])),
                kind=kind, 
                energy_keV=float(feature['energy_keV']),
                beta_scale=beta_scale,
                mu_ch=float(outcome.x[1]), 
                mu_stat_ch=mu_stat,
                width_ch=width, 
                deviance=chi2,
                pearson_chi2=pearson_chi2,
                reduced_pearson_chi2=pearson_chi2/dof if dof > 0 else float('nan'),
                dof=dof,
                reduced_deviance=chi2/dof if dof > 0 else float('nan'),
                bound_hit=bool(bound_hit), 
                x=x, 
                y=y, 
                prediction=pred,
                params=outcome.x,
                fixed_width=fixed_width,
                fixed_beta_scale=fixed_beta_scale,
                center=center, 
                slope_guess=slope_guess,
                background_order=order)


def calibration_run_label(spectrum, run_type):
    """Make a readable title using the calibration filename and MCA live time."""
    info = parse_run_name(spectrum['path'])
    date = info.get('date') if info else None

    if not date:
        date = spectrum.get('measurement_date', 'Unknown date')

    heading = f"Ba-133 {run_type.title()} Calibration ({date})"
    return heading, f"Live time: {spectrum['live_time_s']:,.0f} s"


def diagnostic_plot(spectrum, fitted, output_path, run_type='overnight'):
    """Plot raw bins, fitted bins, signal plus background, and Poisson residuals."""
    kind_labels = {
        'gaussian': ('Gaussian photopeak', r'\mu_{53}'),
        'cs_k_complex': ('Cs K\u03b1/K\u03b2 multiplet', r'\mu_{\alpha 1}'),
        'doublet': ('79.6/81.0 keV double Gaussian', r'\mu_{81}'),
        'edge': ('Gaussian-broadened Compton edge', r'n_{\mathrm{edge}}'),
    }
    model_name, centroid_symbol = kind_labels[fitted['kind']]

    x = fitted['x']
    y = fitted['y']
    pred = fitted['prediction']
    lower, upper = float(x.min()), float(x.max())
    padding = max(6, 0.28 * (upper - lower))

    raw_ch = np.asarray(spectrum['channels'])
    raw_ct = np.asarray(spectrum['counts'])
    mask = (raw_ch >= lower - padding) & (raw_ch <= upper + padding)
    dense = np.linspace(lower, upper, 500)

    width = fitted.get('fixed_width')
    beta_scale = fitted.get('fixed_beta_scale')
    signal, bg = model_components(
        dense, fitted['params'], fitted['kind'], fitted['center'],
        fitted['slope_guess'], width, fitted['background_order'], beta_scale
    )

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    ax.step(raw_ch[mask], raw_ct[mask], where='mid',
            color='0.72', lw=1.1, label='Raw spectrum', zorder=1)
    ax.axvspan(lower, upper, color='0.9', alpha=0.48, zorder=0)
    ax.errorbar(x, y, yerr=np.sqrt(np.maximum(y, 1)), fmt='o',
                markersize=3.1, color='black', ecolor='black',
                elinewidth=0.7, capsize=0, label='Data in fit region', zorder=4)
    ax.plot(dense, signal + bg, color='cornflowerblue', lw=1.5,
            label=model_name, zorder=3)
    ax.plot(dense, bg, '--', color='mediumpurple', lw=1.5,
            label=f'Exponential background (order {fitted["background_order"]})', zorder=2)

    mu = fitted['mu_ch']
    ax.axvline(mu, color='black', ls='--', lw=1.7, alpha=0.8,
               label='Fitted channel', zorder=2)
    stat = fitted['mu_stat_ch']
    win = fitted.get('mu_window_sys_ch')
    chi2dof = fitted.get('reduced_deviance', float('nan'))

    annotation = rf'${centroid_symbol} = {mu:.3f} \pm {stat:.3f}_{{\mathrm{{stat}}}}$ channels'
    if win is not None and np.isfinite(win):
        annotation += '\n' + rf'$\sigma_{{\mathrm{{Window}}}}$: {win:.3f} channels'
    annotation += '\n' + rf'$\chi^2_{{\mathrm{{Poisson}}}}/\mathrm{{dof}} = {chi2dof:.2f}$'

    text_y = 0.04 if fitted['kind'] == 'edge' else 0.975
    text_va = 'bottom' if fitted['kind'] == 'edge' else 'top'
    ax.text(0.015, text_y, annotation, transform=ax.transAxes,
            va=text_va, fontsize=9, bbox={'facecolor': 'white',
                                       'edgecolor': '0.78', 'alpha': 0.95,
                                       'boxstyle': 'round,pad=0.4'})
    ax.set_ylabel('Counts per channel')
    ax.legend(loc='upper right', fontsize=8.5, frameon=False)

    residual = poisson_deviance_residuals(y, pred)
    ax_res.plot(x, residual, 'o', markersize=3.0, color='black')
    ax_res.axhline(0, color='0.5', ls='--', lw=1.0)
    ax_res.set(xlabel='MCA channel', ylabel='Deviance residuals')
    ax_res.set_xlim(lower - padding, upper + padding)

    title, live_time = calibration_run_label(spectrum, run_type)

    subtitle = f'{fitted["display_name"]}, {live_time}'
    fig.suptitle(f'{title}\n{subtitle}', fontsize=12, y=0.985, linespacing=1.5)
    fig.subplots_adjust(top=0.855, bottom=0.11, left=0.13, right=0.97)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def serializable_fit(fitted):
    return {k: v for k, v in fitted.items() if k not in
            {'x', 'y', 'prediction', 'params', 'center', 'slope_guess', 'background_order', 'fixed_width', 'fixed_beta_scale'}}

def fit_calibration_line(feature_rows):
    """
    Weighted line fit to N(E) = alpha*E + beta, inverted to get E = a*N + b
    """
    if len(feature_rows) < 2:
        raise ValueError('At least TWO independently identified features needed')
    
    ch = np.array([r['mu_ch'] for r in feature_rows], dtype=float)
    E = np.array([r['energy_keV'] for r in feature_rows], dtype=float)
    sig = np.array([r['mu_stat_ch'] for r in feature_rows], dtype=float)

    if len(np.unique(E)) < 2 or np.any(~np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError('Energies must be distinct and uncertainties finite/positive')
    
    X = np.column_stack([E, np.ones_like(E)])
    weighted_X = X / sig[:, None]
    weighted_y = ch / sig

    try:
        alpha, beta = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)[0]
        covariance_forward = np.linalg.inv(weighted_X.T @ weighted_X)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Calibration feature energies cannot constrain the line') from exc
    
    if alpha <= 0:
        raise ValueError('Fitted channel-vs-energy slope is not positive')
    
    a, b = 1/alpha, -beta/alpha
    jac = np.array([[-1/alpha**2, 0], [beta/alpha**2, -1/alpha]])
    cov = jac @ covariance_forward @ jac.T
    chi2 = float(np.sum(((ch-(alpha*E+beta))/sig)**2))

    return dict(slope_keV_per_channel=float(a), 
                offset_keV=float(b),
                slope_err_keV_per_channel=float(np.sqrt(max(cov[0,0],0))),
                offset_err_keV=float(np.sqrt(max(cov[1,1],0))),
                covariance_keV=np.asarray(cov).tolist(),
                chi2=chi2, 
                dof=len(ch)-2, 
                n_features=len(ch),
                residuals_keV=(E-(a*ch+b)).tolist())

def inspect(spe_path, out):
    s = read_spe(spe_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.step(s['channels'], s['counts'], where='mid', lw=.9, color="midnightblue")
    ax.set(xlabel='MCA channel', 
           ylabel='Counts/channel', 
           yscale='symlog',
           title=f"{s['path'].name} (live time {s['live_time_s']:.0f} s)")
    
    fig.tight_layout()
    fig.savefig(out/'overview.png', dpi=180)

    plt.close(fig)

    for hi in (250, 750):
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = s['channels'] <= hi
        ax.step(s['channels'][mask], s['counts'][mask], where='mid', lw=.9, color="midnightblue")
        ax.set(xlabel='MCA channel', 
               ylabel='Counts/channel', 
               yscale='symlog',
               title=f'{s["path"].name}: channels 0-{hi}')
        fig.tight_layout()
        fig.savefig(out/f'overview_0_to_{hi}.png', dpi=180)

        plt.close(fig)
    print('Saved whole spectrum and two zoomed overviews.')

def run_overnight(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    s = read_spe(args.spe)

    config = json.loads(Path(args.config).read_text())

    active = [f for f in config['features'] if f.get('enabled', False)]

    rows = []
    for f in active:
        try:
            base = fit_feature(s, f, args.slope_guess)
            shift = []
            for trial_window in f.get('fit_windows', []):
                alt_f = dict(f, window=trial_window)
                try:
                    alt = fit_feature(s, alt_f, args.slope_guess)
                    shift.append(alt['mu_ch'] - base['mu_ch'])
                except (ValueError, RuntimeError):
                    warnings.warn(f"Window {trial_window} failed for {f['name']}")
            # largest observed sensitivity from a window choice

            window_sys = max([abs(v) for v in shift] + [0])
            extra_sys = float(f.get('systematic_centroid_ch', 0))
            sys_ch = float(np.hypot(window_sys, extra_sys))

            row = serializable_fit(base)

            row['mu_window_sys_ch'] = float(window_sys)
            row['mu_extra_sys_ch'] = extra_sys
            row['mu_sys_ch'] = sys_ch
            row['mu_total_ch'] = float(np.hypot(base['mu_stat_ch'], sys_ch))
            row['fit_windows_ch'] = f.get('fit_windows', [])
            row['window_shifts_ch'] = shift

            base['mu_window_sys_ch'] = window_sys
            base['mu_total_ch'] = row['mu_total_ch']

            rows.append(row)

            diagnostic_plot(s, base, out/f"feature_{f['name']}.png", run_type='overnight')

            print(f"{f['name']}: {row['mu_ch']:.3f} ± {row['mu_stat_ch']:.3f} (stat)"
                  f" ± {sys_ch:.3f} (window) channels; dev/dof="
                  f"{row['reduced_deviance']:.2f}; bound={row['bound_hit']}")
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Skipping {f['name']}: {exc}")

    pd.DataFrame(rows).to_csv(out/'overnight_features.csv', index=False)
    result = dict(reference_spectrum=str(s['path']), live_time_s=s['live_time_s'],
                  slope_guess=float(args.slope_guess), features=rows)
    
    if len(rows) >= 2:
        try:
            result['linear_calibration'] = fit_calibration_line(rows)
            print('Calibration E(keV) = a*channel + b:', result['linear_calibration'])

            line = result['linear_calibration']
            fig, ax = plt.subplots(figsize=(6.8, 4.5))
            ch = np.array([r['mu_ch'] for r in rows])
            en = np.array([r['energy_keV'] for r in rows])
            er = np.array([r['mu_total_ch'] for r in rows])

            ax.errorbar(ch, en, xerr=er, fmt='o', markersize=3,
                        capsize=4, capthick=1.1, elinewidth=1.3,
                        color="midnightblue", ecolor="midnightblue",
                        label='Calibration points', zorder=5)

            for point, row in zip(zip(ch, en), rows):
                ax.annotate(row.get('plot_label', row['name']), point,
                            xytext=(-3, 10), textcoords='offset points',
                            ha='right', va='bottom', fontsize=8, color='0.25')

            xv = np.linspace(0, max(ch)+10, 200)
            ax.plot(xv, line['slope_keV_per_channel']*xv+line['offset_keV'],
                    color='cornflowerblue', lw=0.8, label='Linear fit', zorder=3)

            cov = np.array(line['covariance_keV'])

            energy_fit = (line['slope_keV_per_channel'] * xv + line['offset_keV'])
            energy_err = np.sqrt(cov[0, 0] * xv**2 + 2 * cov[0, 1] * xv + cov[1, 1])

            ax.fill_between(
                xv,
                energy_fit - energy_err,
                energy_fit + energy_err,
                color='cornflowerblue',
                alpha=0.45,
                zorder=2,
                label=r'$1\sigma$ fit uncertainty'
            )

            heading, live_time = calibration_run_label(s, 'overnight')
            ax.set(xlabel='MCA channel', ylabel='Deposited energy (keV)', title=f'{heading}\n{live_time}')
            ax.set_xlim(0, max(ch) + 15)
            ax.set_ylim(0, max(en) + 30)
            ax.legend(frameon=False, loc='lower right', fontsize=8)

            if line['dof'] > 0:
                ax.text(
                    0.03, 0.97,
                    (
                        rf'$\chi^2/\mathrm{{dof}} = {line["chi2"]/line["dof"]:.3f}$'
                        '\n'
                        rf'$a = ({line["slope_keV_per_channel"]:.3f} \pm '
                        rf'{line["slope_err_keV_per_channel"]:.3f})$ keV/channel'
                        '\n'
                        rf'$b = ({line["offset_keV"]:.3f} \pm '
                        rf'{line["offset_err_keV"]:.3f})$ keV'
                    ),
                    transform=ax.transAxes,
                    va='top',
                    fontsize=9,
                    bbox=dict(
                        facecolor='white',
                        edgecolor='0.8',
                        boxstyle='round,pad=0.35',
                        alpha=0.95
                    )
                )

            fig.tight_layout()
            fig.savefig(out/'overnight_calibration_line.png', dpi=180)
            plt.close(fig)

        except ValueError as exc:
            warnings.warn(str(exc))
    else:
        print('Only one verified feature: no measured slope/offset can be calculated yet.')
    (out/'overnight_calibration.json').write_text(json.dumps(result, indent=2))

def run_daily(args):
    """Fit a separate energy-to-channel calibration for each daily spectrum."""
    reference = json.loads(Path(args.reference).read_text())
    if 'linear_calibration' not in reference:
        raise ValueError('Overnight reference has no multipoint calibration.')

    overnight = reference['linear_calibration']
    reference_slope = overnight['slope_keV_per_channel']

    features = reference['features']
    cs_ref = next((f for f in features if f['kind'] == 'cs_k_complex'), None)
    ba_ref = next((f for f in features if f['kind'] == 'doublet'), None)
    if cs_ref is None or ba_ref is None:
        raise ValueError('Overnight reference must contain the Cs K and 81 keV fits.')

    if cs_ref.get('beta_scale') is None:
        raise ValueError('Cs K fit needs an overnight K-beta amplitude scale.')

    if not np.isclose(cs_ref['energy_keV'], CS_K_ALPHA1, atol=0.001):
        raise ValueError('Overnight Cs K-alpha1 reference must use 30.97313 keV.')

    scale = getattr(args, 'channel_scale', 1.0)
    shift = getattr(args, 'channel_shift', 0.0)

    if scale <= 0:
        raise ValueError('--channel-scale must be positive.')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    patterns = args.files if isinstance(args.files, list) else [args.files]
    paths = sorted(set(path for pattern in patterns
                       for path in glob.glob(pattern, recursive=True)))
    if not paths:
        raise FileNotFoundError(f'No files matched {patterns}')

    calibration_rows = []
    feature_rows = []

    def fit_daily_peak(spectrum, ref, slope, channel_shift=0.0, shrink=0):
        half_widths = {
            'cs_k_complex': (13, 19),
            'doublet': (18, 18),
            'gaussian': (11, 11),
        }

        left, right = half_widths[ref['kind']]
        center = scale * ref['mu_ch'] + shift + channel_shift

        config = {
            'name': ref['name'],
            'display_name': ref.get('display_name', ref['name']),
            'plot_label': ref.get('plot_label', ref.get('display_name', ref['name'])),
            'kind': ref['kind'],
            'energy_keV': ref['energy_keV'],
            'expected_channel': center,
            'width_guess': scale * ref['width_ch'],
            'background_order': 1,
            'window': [max(0, center - scale * left + shrink),
                       center + scale * right - shrink],
        }

        kwargs = {}
        if ref['kind'] == 'cs_k_complex':
            kwargs['fixed_beta_scale'] = cs_ref['beta_scale']

        return fit_feature(spectrum, config, slope, **kwargs)

    for path in paths:
        if Path(path).resolve() == Path(reference['reference_spectrum']).resolve():
            continue

        spectrum = read_spe(path)
        name = Path(path).stem
        fitted = []

        run_info = parse_run_name(spectrum['path'])
        date = (run_info or {}).get('date') or spectrum.get('measurement_date')

        row = {
            'file': str(path),
            'date': date,
            'live_time_s': spectrum['live_time_s'],
            'method': 'daily_independent_multipoint',
        }

        try:
            slope = reference_slope / scale
            ba_first = fit_daily_peak(spectrum, ba_ref, slope)
            anchor_shift = ba_first['mu_ch'] - (scale * ba_ref['mu_ch'] + shift)

            # Update the line separations as the daily slope is refined.
            for iteration in range(2):
                fitted = []

                for ref in [cs_ref, ba_ref]:
                    try:
                        fit = fit_daily_peak(spectrum, ref, slope, anchor_shift)
                        window_shifts = []
                        for shrink in (2, 4, -2):
                            try:
                                alt = fit_daily_peak(spectrum, ref, slope, anchor_shift, shrink=shrink)
                                window_shifts.append(alt['mu_ch'] - fit['mu_ch'])
                            except (ValueError, RuntimeError):
                                warnings.warn(f'{name}: alternate window failed for {ref["name"]}')

                        window_unc = max([abs(v) for v in window_shifts] + [0])
                        reference_unc = scale * ref.get('mu_window_sys_ch', 0.0)

                        fit['mu_window_sys_ch'] = float(window_unc)
                        fit['mu_reference_window_sys_ch'] = float(reference_unc)
                        fit['window_shifts_ch'] = window_shifts
                        fit['mu_total_ch'] = float(np.sqrt(fit['mu_stat_ch']**2 +
                                                          window_unc**2 + reference_unc**2))

                        width_ratio = fit['width_ch'] / (scale * ref['width_ch'])
                        # Reject unconstrained fits before calibrating
                        is_valid = (not fit['bound_hit']
                                    and fit['reduced_deviance'] < 5.0
                                    and np.isfinite(fit['mu_total_ch'])
                                    and fit['mu_stat_ch'] < 3.0 * scale
                                    and 0.4 < width_ratio < 2.6)

                        if ref['kind'] == 'gaussian':
                            is_valid &= (0.65 < width_ratio < 1.7
                                         and window_unc < 1.0 * scale)

                        fit['use_in_calibration'] = bool(is_valid)
                        fitted.append(fit)

                    except (ValueError, RuntimeError) as exc:
                        warnings.warn(f'{name}: {ref["name"]} fit failed: {exc}')

                usable = [f for f in fitted if f['use_in_calibration']]
                if len(usable) < 2:
                    break

                daily_points = [dict(f, mu_stat_ch=f['mu_total_ch']) for f in usable]
                line = fit_calibration_line(daily_points)
                slope = line['slope_keV_per_channel']

            for fit in fitted:
                entry = serializable_fit(fit)
                entry.update(file=str(path), use_in_calibration=fit['use_in_calibration'])
                feature_rows.append(entry)

                diagnostic_plot(spectrum, fit, out / f'{name}_{fit["name"]}.png', run_type='daily')

            if len(usable) < 2:
                row.update(status='INSUFFICIENT_FEATURES', n_features=len(usable))
            else:
                row.update(line)
                row.update(status='PROVISIONAL',
                           gain_ratio_vs_overnight=reference_slope / slope,
                           used_features='; '.join(f['name'] for f in usable))

                ch = np.array([f['mu_ch'] for f in usable])
                en = np.array([f['energy_keV'] for f in usable])
                er = np.array([f['mu_total_ch'] for f in usable])

                fig, ax = plt.subplots(figsize=(6.8, 4.5))
                ax.errorbar(ch, en, xerr=er, fmt='o', markersize=3,
                            capsize=4, capthick=1.1, elinewidth=1.3,
                            color='midnightblue', ecolor='midnightblue',
                            label='Calibration points', zorder=5)

                for point, fit in zip(zip(ch, en), usable):
                    ax.annotate(fit.get('plot_label', fit['name']), point,
                                xytext=(-3, 10), textcoords='offset points',
                                ha='right', va='bottom', fontsize=8, color='0.25')

                xv = np.linspace(0, max(ch) + 10, 200)
                energy_fit = line['slope_keV_per_channel'] * xv + line['offset_keV']
                ax.plot(xv, energy_fit, color='cornflowerblue', lw=0.8,
                        label='Linear fit', zorder=3)

                cov = np.array(line['covariance_keV'])
                energy_err = np.sqrt(np.maximum(
                    cov[0, 0] * xv**2 + 2 * cov[0, 1] * xv + cov[1, 1], 0
                ))

                ax.fill_between(xv, energy_fit - energy_err, energy_fit + energy_err,
                                color='cornflowerblue', alpha=0.45, zorder=2,
                                label=r'$1\sigma$ fit uncertainty')

                heading, live_time = calibration_run_label(spectrum, 'daily')
                ax.set(xlabel='MCA channel', ylabel='Deposited energy (keV)',
                       title=f'{heading}\n{live_time}')
                
                ax.set_xlim(0, max(ch) + 15)
                ax.set_ylim(0, max(en) + 30)

                ax.legend(frameon=False, loc='lower right', fontsize=8)

                ax.text(
                    0.03, 0.97,
                    (
                        rf'$a = ({line["slope_keV_per_channel"]:.6f} \pm '
                        rf'{line["slope_err_keV_per_channel"]:.6f})$ keV/channel'
                        '\n'
                        rf'$b = ({line["offset_keV"]:.4f} \pm '
                        rf'{line["offset_err_keV"]:.4f})$ keV'
                    ),
                    transform=ax.transAxes,
                    va='top',
                    fontsize=9,
                    bbox=dict(
                        facecolor='white',
                        edgecolor='0.8',
                        boxstyle='round,pad=0.35',
                        alpha=0.95
                    )
                )

                fig.tight_layout()
                fig.savefig(out / f'{name}_calibration_line.png', dpi=180)
                plt.close(fig)

                daily_result = {
                    'file': str(path),
                    'date': date,
                    'live_time_s': spectrum['live_time_s'],
                    'linear_calibration': line,
                    'features': [serializable_fit(f) for f in usable],
                }
                (out / f'{name}_calibration.json').write_text(
                    json.dumps(daily_result, indent=2)
                )

                print(f'{name}: a = {slope:.6f} keV/channel, '
                      f'b = {line["offset_keV"]:.3f} keV '
                      f'({len(usable)} features)')

        except (ValueError, RuntimeError) as exc:
            row.update(status='FAILED', problem=str(exc))
            warnings.warn(f'{name}: {exc}')

        calibration_rows.append(row)

    pd.DataFrame(feature_rows).to_csv(out / 'daily_features.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(out / 'daily_calibrations.csv', index=False)

    print(f'Saved results to {out}; review diagnostics before using any calibration.')

def main():
    p = argparse.ArgumentParser(description=__doc__)

    sub = p.add_subparsers(dest='mode', required=True)

    t = sub.add_parser('inspect')
    t.add_argument('--spe', required=True)
    t.add_argument('--output', required=True)

    t = sub.add_parser('overnight')
    t.add_argument('--spe', required=True)
    t.add_argument('--config', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--slope-guess', type=float, default=.56)

    t = sub.add_parser('daily')
    t.add_argument('--reference', required=True)
    t.add_argument('--files', nargs='+', required=True,
                   help='One or more .Spe files or quoted glob patterns')
    t.add_argument('--output', required=True)
    t.add_argument('--channel-scale', type=float, default=1.0)
    t.add_argument('--channel-shift', type=float, default=0.0)

    args = p.parse_args()
    
    if args.mode == 'inspect':
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        inspect(args.spe, out)
    elif args.mode == 'overnight':
        run_overnight(args)
    else:
        run_daily(args)


if __name__ == '__main__':
    main()
