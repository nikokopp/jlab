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

def paper_calibration_figure(spectrum, fitted_features, output_dir):
    """Publication figure: overview spectrum plus four boxed fitted features.

    Reuses the fitted parameter arrays from fit_feature (no refitting).
    Requires four successfully fitted overnight features.
    AI-assisted figure layout; verify all features and caption independently.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if len(fitted_features) != 4:
        warnings.warn(
            f"Composite paper figure requires four fitted features; "
            f"got {len(fitted_features)}. Skipping."
        )
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fitted_features = sorted(fitted_features, key=lambda fit: fit['mu_ch'])
    letters = ['A', 'B', 'C', 'D']

    raw_color = '0.72'
    point_color = 'black'
    fit_color = 'cornflowerblue'

    def energy_label(fit):
        energy = fit['energy_keV']
        if energy >= 100:
            return f'{energy:.0f} keV'
        if energy >= 10:
            return f'{energy:.0f} keV'
        return f'{energy:.1f} keV'

    def count_tick_label(y, _pos):
        if abs(y) >= 1e6:
            return f'{y / 1e6:g}M'
        if abs(y) >= 1e3:
            return f'{y / 1e3:g}k'
        return f'{y:g}'

    plot_style = {
        'font.family': 'sans-serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'savefig.facecolor': 'white',
    }

    with plt.rc_context(plot_style):
        fig = plt.figure(figsize=(7.2, 4.45))
        gs = fig.add_gridspec(
            2, 4, height_ratios=(1.28, 1.0),
            left=0.085, right=0.985, bottom=0.14, top=0.93,
            hspace=0.26, wspace=0.26,
        )
        ax_main = fig.add_subplot(gs[0, :])
        zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

        channels = np.asarray(spectrum['channels'], dtype=float)
        counts = np.asarray(spectrum['counts'], dtype=float)

        ax_main.step(channels, counts, where='mid', lw=1.0,
                     color=raw_color, zorder=1)
        ax_main.set_yscale('symlog', linthresh=1.0)
        ax_main.set_xlim(float(channels.min()), float(channels.max()))
        ax_main.set_ylim(0, max(float(np.max(counts)) * 1.25, 10.0))
        ax_main.set_ylabel('Counts per channel')
        ax_main.tick_params(axis='both', length=3.0, width=0.7)

        legend_handles = [
            Line2D([], [], color=raw_color, lw=1.0, label='Raw spectrum'),
            Line2D([], [], color=point_color, marker='o', linestyle='None',
                   markersize=3.5, label='Data in fit region'),
            Line2D([], [], color=fit_color, lw=1.5, label='Fitted line'),
        ]
        ax_main.legend(handles=legend_handles, fontsize=7.5, ncol=1,
                       loc='upper right', frameon=False,
                       handlelength=2.0, borderpad=0.15, labelspacing=0.35)

        for i, (fit, ax_zoom) in enumerate(zip(fitted_features, zoom_axes)):
            x = np.asarray(fit['x'], dtype=float)
            y = np.asarray(fit['y'], dtype=float)
            lower, upper = float(x.min()), float(x.max())
            dense = np.linspace(lower, upper, 600)
            signal, background = model_components(
                dense,
                fit['params'], fit['kind'], fit['center'],
                fit['slope_guess'], fit.get('fixed_width'),
                fit['background_order'], fit.get('fixed_beta_scale'),
            )
            total = signal + background

            # bottom feature panel
            ax_zoom.step(x, y, where='mid', lw=1.0, color=raw_color, zorder=1)
            ax_zoom.plot(x, y, 'o', ms=3.0, color=point_color, zorder=2)
            ax_zoom.plot(dense, total, color=fit_color, lw=1.5, zorder=3)
            ax_zoom.set_xlim(lower, upper)
            ax_zoom.set_ylim(0, 1.12 * max(float(y.max()), float(total.max()), 1.0))
            ax_zoom.set_title(f'({letters[i]}) {energy_label(fit)}', pad=2.5)
            ax_zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_zoom.yaxis.set_major_formatter(FuncFormatter(count_tick_label))
            ax_zoom.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax_zoom.tick_params(axis='both', length=2.5, width=0.65,
                                pad=2.2, labelsize=6.7)
            for spine in ax_zoom.spines.values():
                spine.set_linewidth(0.7)

            # matching boxed region on overview spectrum
            raw_mask = (channels >= lower) & (channels <= upper)
            local_counts = counts[raw_mask] if np.any(raw_mask) else y
            box_bottom = max(0.2, float(np.percentile(local_counts, 3)) * 0.8)
            box_top = max(float(np.max(local_counts)) * 1.03, box_bottom + 1.0)
            box = Rectangle((lower, box_bottom), upper - lower, box_top - box_bottom,
                            facecolor='none', edgecolor='black', linewidth=1.0, zorder=4)
            ax_main.add_patch(box)
            ax_main.text(lower + 0.10 * (upper - lower), box_top / 1.03,
                         f'({letters[i]})',
                         ha='left', va='top', fontsize=7.5,
                         bbox=dict(facecolor='white', edgecolor='none', pad=0.3, alpha=0.95),
                         zorder=5)

        zoom_axes[0].set_ylabel('Counts / channel', fontsize=7.6)
        fig.supxlabel('MCA channel', y=0.055, fontsize=8)

        basename = output_dir / 'ba133_spectrum_fits'
        fig.savefig(basename.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.025)
        fig.savefig(basename.with_suffix('.png'), dpi=350,
                    bbox_inches='tight', pad_inches=0.025)
        plt.close(fig)
        print(f'Saved paper figure: {basename}.pdf and {basename}.png')

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
    paper_fits = []
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
            paper_fits.append(base)

            diagnostic_plot(s, base, out/f"feature_{f['name']}.png", run_type='overnight')

            print(f"{f['name']}: {row['mu_ch']:.3f} ± {row['mu_stat_ch']:.3f} (stat)"
                  f" ± {sys_ch:.3f} (window) channels; dev/dof="
                  f"{row['reduced_deviance']:.2f}; bound={row['bound_hit']}")
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f"Skipping {f['name']}: {exc}")

    # Save a paper-ready composite separately from the full diagnostics.
    # With the script inside relativistic_dynamics/analysis/ this resolves
    # to relativistic_dynamics/paper_figures/.
    paper_dir = (Path(args.paper_dir) if args.paper_dir else
                 Path(__file__).resolve().parent.parent / 'paper_figures')
    paper_calibration_figure(s, paper_fits, paper_dir)

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
    t.add_argument('--paper-dir', default=None,
                   help='Folder for composite paper figure (default: ../paper_figures relative to analysis/)')

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


# ======================================================================
# ======================================================================
#
#                    PAPER FIGURES: MCA CALIBRATION
#
#   1) Composite figure:
#      - top: overnight raw spectrum with feature boxes (a)-(d)
#      - bottom: the 4 individual feature fits in a 2x2 layout
#
#   2) Separate linear calibration figure:
#      - deposited energy vs fitted channel
#
# ======================================================================
# ======================================================================

if __name__ == '__main__':

    # Small parser just to recover the arguments used for the overnight run
    paper_parser = argparse.ArgumentParser(add_help=False)
    paper_parser.add_argument('mode', nargs='?')
    paper_parser.add_argument('--spe')
    paper_parser.add_argument('--config')
    paper_parser.add_argument('--output')
    paper_parser.add_argument('--slope-guess', type=float, default=0.56)

    paper_args, _ = paper_parser.parse_known_args()

    # Only make these paper figures for overnight calibration runs
    if (
        paper_args.mode == 'overnight'
        and paper_args.spe is not None
        and paper_args.config is not None
        and paper_args.output is not None
    ):
        from matplotlib.patches import Rectangle

        paper_dir = Path("/home/rck67/jlab/relativistic_dynamics/figures/paper_figures")
        paper_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Reload the spectrum and refit the active overnight features
        # ------------------------------------------------------------------
        spectrum = read_spe(paper_args.spe)
        config = json.loads(Path(paper_args.config).read_text())
        active = [f for f in config['features'] if f.get('enabled', False)]

        rows = []
        fitted_objects = []

        for f in active:
            try:
                base = fit_feature(spectrum, f, paper_args.slope_guess)

                shifts = []
                for trial_window in f.get('fit_windows', []):
                    alt_f = dict(f, window=trial_window)
                    try:
                        alt = fit_feature(spectrum, alt_f, paper_args.slope_guess)
                        shifts.append(alt['mu_ch'] - base['mu_ch'])
                    except (ValueError, RuntimeError):
                        warnings.warn(f"Window {trial_window} failed for {f['name']}")

                window_sys = max([abs(v) for v in shifts] + [0])
                extra_sys = float(f.get('systematic_centroid_ch', 0))
                sys_ch = float(np.hypot(window_sys, extra_sys))

                row = serializable_fit(base)
                row['mu_window_sys_ch'] = float(window_sys)
                row['mu_extra_sys_ch'] = extra_sys
                row['mu_sys_ch'] = sys_ch
                row['mu_total_ch'] = float(np.hypot(base['mu_stat_ch'], sys_ch))
                row['fit_windows_ch'] = f.get('fit_windows', [])
                row['window_shifts_ch'] = shifts

                # preserve window and plotting metadata for the paper figure
                row['window'] = f['window']
                row['expected_channel'] = f.get('expected_channel', None)

                base['mu_window_sys_ch'] = window_sys
                base['mu_total_ch'] = row['mu_total_ch']
                base['window'] = f['window']

                rows.append(row)
                fitted_objects.append(base)

            except (ValueError, RuntimeError) as exc:
                warnings.warn(f"Skipping {f['name']}: {exc}")

        if len(rows) < 4:
            raise ValueError(
                f"Need at least 4 successfully fitted overnight features for the composite figure; found {len(rows)}."
            )

        # Sort left-to-right in channel so labels (a)-(d) match the spectrum order
        sort_order = np.argsort([r['mu_ch'] for r in rows])
        rows = [rows[i] for i in sort_order]
        fitted_objects = [fitted_objects[i] for i in sort_order]

        # If more than 4 are active, use the first 4 in channel order for the composite panel figure
        rows4 = rows[:4]
        fitted4 = fitted_objects[:4]
        panel_letters = ['a', 'b', 'c', 'd']

        # ------------------------------------------------------------------
        # Common formatting
        # ------------------------------------------------------------------
        label_size = 12
        tick_size = 10
        legend_size = 9
        stats_size = 8.7
        panel_size = 14

        raw_ch = np.asarray(spectrum['channels'], dtype=float)
        raw_ct = np.asarray(spectrum['counts'], dtype=float)

# ======================================================================
# ======================================================================
#
#             FIGURE 1: OVERNIGHT MCA CALIBRATION
#
# ======================================================================
# ======================================================================

    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    fig_cal = plt.figure(figsize=(13.2, 6.0))

    gs = fig_cal.add_gridspec(
        2, 4,
        height_ratios=[1.05, 1.15],
        hspace=0.32,
        wspace=0.16
    )

    # Full spectrum
    ax_top = fig_cal.add_subplot(gs[0, :])

    # Four feature fits in one row
    feature_axes = [
        fig_cal.add_subplot(gs[1, i])
        for i in range(4)
    ]

    fig_cal.subplots_adjust(
        left=0.085,
        right=0.990,
        top=0.975,
        bottom=0.18
    )

    raw_ch = np.asarray(spectrum["channels"], dtype=float)
    raw_ct = np.asarray(spectrum["counts"], dtype=float)


    # ======================================================================
    # TOP PANEL: RAW OVERNIGHT CALIBRATION SPECTRUM
    # ======================================================================

    # Only display channels 0-800
    spectrum_mask = (raw_ch >= 0) & (raw_ch <= 800)

    ax_top.step(
        raw_ch[spectrum_mask],
        raw_ct[spectrum_mask],
        where="mid",
        lw=1.0,
        color="midnightblue"
    )

    ax_top.set_xlabel(
        "MCA channel",
        fontsize=11
    )

    ax_top.set_ylabel(
        "Counts per channel",
        fontsize=11
    )

    ax_top.set_xlim(0, 800)
    ax_top.set_yscale("symlog", linthresh=1)

    ax_top.tick_params(labelsize=9)
    ax_top.grid(alpha=0.12)


    # ----------------------------------------------------------------------
    # Box the four fitted features
    # ----------------------------------------------------------------------

    box_tops = []

    for letter, row in zip(panel_letters, rows4):

        lower, upper = map(float, row["window"])

        mask = (raw_ch >= lower) & (raw_ch <= upper)

        if not np.any(mask):
            raise ValueError(
                f"No spectrum bins found in window {lower}-{upper}"
            )

        local_counts = raw_ct[mask]
        positive = local_counts[local_counts > 0]

        if len(positive) > 0:
            y0 = max(
                1.0,
                np.percentile(positive, 5) * 0.7
            )
        else:
            y0 = 1.0

        y1 = max(
            local_counts.max() * 1.20,
            y0 * 2.0
        )

        box_tops.append(y1)

        rect = Rectangle(
            (lower, y0),
            upper - lower,
            y1 - y0,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            zorder=6
        )

        ax_top.add_patch(rect)

        # Bold labels above boxes
        ax_top.annotate(
            f"({letter})",
            xy=((lower + upper) / 2, y1),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            zorder=7
        )

    # Additional space at top for labels
    ax_top.set_ylim(
        bottom=0,
        top=max(
            np.max(raw_ct[spectrum_mask]),
            max(box_tops)
        ) * 2.0
    )


    # ======================================================================
    # BOTTOM ROW: FOUR INDIVIDUAL FEATURE FITS
    # ======================================================================

    centroid_symbols = {
        "gaussian": r"\mu",
        "cs_k_complex": r"\mu_{\alpha 1}",
        "doublet": r"\mu_{81}",
        "edge": r"n_{\mathrm{edge}}"
    }

    for ax, fitted, row, letter in zip(
        feature_axes,
        fitted4,
        rows4,
        panel_letters
    ):

        x = np.asarray(fitted["x"], dtype=float)
        y = np.asarray(fitted["y"], dtype=float)

        lower = float(x.min())
        upper = float(x.max())

        # Keep the zoom focused on the fitted feature
        padding = max(4, 0.20 * (upper - lower))

        x_lower = max(raw_ch.min(), lower - padding)
        x_upper = min(raw_ch.max(), upper + padding)

        mask = (
            (raw_ch >= x_lower)
            & (raw_ch <= x_upper)
        )

        dense = np.linspace(lower, upper, 500)

        # Reconstruct the fitted model
        signal, bg = model_components(
            dense,
            fitted["params"],
            fitted["kind"],
            fitted["center"],
            fitted["slope_guess"],
            fitted.get("fixed_width"),
            fitted["background_order"],
            fitted.get("fixed_beta_scale")
        )

        total_model = signal + bg


        # ------------------------------------------------------------------
        # Fitted region
        # ------------------------------------------------------------------

        ax.axvspan(
            lower,
            upper,
            color="0.92",
            alpha=0.45,
            zorder=0
        )


        # ------------------------------------------------------------------
        # Raw spectrum
        # ------------------------------------------------------------------

        ax.step(
            raw_ch[mask],
            raw_ct[mask],
            where="mid",
            color="0.72",
            linewidth=0.9,
            zorder=1
        )


        # ------------------------------------------------------------------
        # Data points used in the fit
        # Smaller markers to make uncertainties more visible
        # ------------------------------------------------------------------

        ax.errorbar(
            x, y,
            yerr=np.sqrt(np.maximum(y, 1)),
            fmt="o",
            markersize=1.8,
            color="black",
            ecolor="black",
            elinewidth=0.6,
            capsize=0,
            zorder=5
        )


        # ------------------------------------------------------------------
        # Total fitted model
        # ------------------------------------------------------------------

        ax.plot(
            dense,
            total_model,
            color="cornflowerblue",
            linewidth=1.7,
            zorder=4
        )


        # ------------------------------------------------------------------
        # Background model
        # ------------------------------------------------------------------

        ax.plot(
            dense,
            bg,
            linestyle="--",
            color="mediumpurple",
            linewidth=1.3,
            zorder=3
        )


        # ------------------------------------------------------------------
        # Fitted channel
        # ------------------------------------------------------------------

        mu = fitted["mu_ch"]

        ax.axvline(
            mu,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            zorder=2
        )


        # ==================================================================
        # FORMATTING
        # ==================================================================

        # Only display bold panel letters, without feature names
        ax.set_title(
            f"({letter})",
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=4
        )

        ax.set_xlabel(
            "MCA channel",
            fontsize=10
        )

        ax.set_xlim(x_lower, x_upper)

        # ------------------------------------------------------------------
        # Independently scale each panel
        # ------------------------------------------------------------------

        local_max = max(
            np.max(raw_ct[mask]),
            np.max(total_model),
            np.max(y + np.sqrt(np.maximum(y, 1)))
        )

        # Extra space BELOW zero for panel (a), so the statistics
        # can sit beneath the feature without hiding the data.
        if letter == "a":
            y_bottom = -0.30 * local_max
            y_top = 1.12 * local_max

        elif letter == "b":
            y_bottom = 0.4
            y_top = 1.25 * local_max

        else:
            y_bottom = 0
            y_top = 1.12 * local_max

        ax.set_ylim(y_bottom, y_top)

        # ------------------------------------------------------------------
        # Scientific count scaling
        #
        # Rather than using Matplotlib's automatic 1e5 label
        # above each plot, display scaled tick labels with
        # the multiplier placed beside the y-axis.
        # ------------------------------------------------------------------

        exponent = int(np.floor(np.log10(local_max)))
        count_scale = 10.0 ** exponent

        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, pos, s=count_scale:
                    f"{value/s:g}" if value >= -1e-9 else ""
            )
        )

        ax.set_ylabel(
            rf"$\times 10^{{{exponent}}}$",
            fontsize=8,
            rotation=90,
            labelpad=0
        )

        # Set ticks at readable intervals; no negative count labels
        locator = MaxNLocator(nbins=4)

        y_ticks = locator.tick_values(0, y_top)
        y_ticks = y_ticks[
            (y_ticks >= 0) & (y_ticks <= y_top)
        ]

        ax.set_yticks(y_ticks)

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=4)
        )

        ax.tick_params(
            axis="both",
            labelsize=8,
            pad=2
        )

        ax.grid(alpha=0.13)


        # ==================================================================
        # FIT STATISTICS
        # ==================================================================

        stat = fitted["mu_stat_ch"]
        chi2dof = fitted["reduced_deviance"]
        symbol = centroid_symbols[fitted["kind"]]

        stats_text = (
            rf"${symbol} = {mu:.2f}"
            rf"\pm {stat:.2f}_{{\mathrm{{stat}}}}$"
            "\n"
            rf"$D/\mathrm{{dof}} = {chi2dof:.2f}$"
        )

        # Consistent bottom-left placement
        ax.text(
            0.025,
            0.035,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            linespacing=1.3,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.94,
                "boxstyle": "round,pad=0.3"
            },
            zorder=8
        )


    # ======================================================================
    # SHARED Y-AXIS TITLE FOR BOTTOM PANELS
    # ======================================================================

    # Place shared label centered vertically along bottom row
    bottom_position = feature_axes[0].get_position()

    fig_cal.text(
        0.04,
        bottom_position.y0 + bottom_position.height / 2,
        "Counts per channel",
        va="center",
        ha="center",
        rotation=90,
        fontsize=11
    )


    # ======================================================================
    # SHARED LEGEND
    # ======================================================================

    legend_handles = [
        Line2D(
            [0], [0],
            color="0.72",
            lw=1.2,
            label="Raw spectrum"
        ),
        Line2D(
            [0], [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=3,
            label="Data used in fit"
        ),
        Line2D(
            [0], [0],
            color="cornflowerblue",
            lw=1.8,
            label="Fitted model"
        ),
        Line2D(
            [0], [0],
            color="mediumpurple",
            ls="--",
            lw=1.3,
            label="Background"
        ),
        Line2D(
            [0], [0],
            color="black",
            ls="--",
            lw=1.1,
            label="Fitted channel"
        )
    ]

    fig_cal.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        fontsize=9,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.0
    )


    # ======================================================================
    # SAVE FIGURE
    # ======================================================================

    composite_pdf = (
        paper_dir / "overnight_calibration_composite_paper.pdf"
    )

    composite_png = (
        paper_dir / "overnight_calibration_composite_paper.png"
    )

    fig_cal.savefig(
        composite_pdf,
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig_cal.savefig(
        composite_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"Saved: {composite_pdf}")
    print(f"Saved: {composite_png}")

    plt.close(fig_cal)