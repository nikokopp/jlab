"""
Instructions on using this code for different calibrations:

For overnight calibrations, first inspect the spectrum and decide which spectral features to enable in calibration_features.json.
Then, run the calibration.

For daily calibrations, use the results of the overnight calibration as a reference.

Overnight:
    python mca_calibration_pipeline.py inspect --spe data.Spe --output results
    python mca_calibration_pipeline.py overnight --spe data.Spe \
        --config calibration_features.json --output results
Daily:
    python mca_calibration_pipeline.py daily --reference results/overnight_calibration.json \
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
from spe_analyzer import read_spe

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

def model(x, params, kind, center, slope_guess, fixed_width=None, background_order=1, fixed_beta_scale=None):
    """3 types of feature models: gaussian peak, constrained double gaussian, or Compton edge.

    Local positive background: b0*exp(s1*(x-center) + s2*(x-center)^2).
    Over short windows it resembles a linear/quadratic background, but never results in negative poisson mean.
    """
    amp, mu = params[:2]

    j = 2

    if fixed_width is None:
        width = params[j]
        j += 1
    else:
        width = fixed_width

    log_b0 = params[j]
    slope = params[j + 1]
    curvature = params[j + 2] if background_order == 2 else 0.

    z = x - center

    background = np.exp(np.clip(log_b0 + slope * z + curvature * z**2, -30, 30))
    if kind == 'gaussian':
        signal = amp * np.exp(-0.5 * ((x - mu) / width)**2)
    elif kind == 'cs_k_complex': 
        # blend of a bunch of emission lines, so more complex than just two gaussians
        # mu is the Kalpha1 position, not centroid of entire curve
        # Ka1/Ka2 relative strength fixed from atomic data

        beta_scale = params[-1] if fixed_beta_scale is None else fixed_beta_scale
        signal = np.zeros_like(x, dtype=float)

        for i, (E, rel) in enumerate(zip(CS_K_E, CS_K_REL)):
            center_i = mu + (E - CS_K_ALPHA1) / slope_guess

            if i >= 2:
                rel *= beta_scale
            signal += amp * rel * np.exp(-0.5 * ((x - center_i) / width)**2)

    elif kind == 'doublet': 
        # double gaussian
        delta = (E_81 - E_79) / slope_guess
        signal = amp * (np.exp(-0.5 * ((x - mu) / width)**2)
                        + R_79_81 * np.exp(-0.5 * ((x - (mu - delta)) / width)**2))
        
    elif kind == 'edge':
        # local approximate edge model
        signal = 0.5 * amp * erfc((x - mu) / (np.sqrt(2) * width))

    else:
        raise ValueError(f'Unknown feature kind: {kind}')
    
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
                kind=kind, 
                energy_keV=float(feature['energy_keV']),
                beta_scale=beta_scale,
                mu_ch=float(outcome.x[1]), 
                mu_stat_ch=mu_stat,
                width_ch=width, 
                deviance=chi2, 
                dof=dof,
                reduced_deviance=chi2/dof if dof > 0 else float('nan'),
                bound_hit=bool(bound_hit), 
                x=x, 
                y=y, 
                prediction=pred,
                params=outcome.x, 
                center=center, 
                slope_guess=slope_guess,
                background_order=order)


def diagnostic_plot(spectrum, fitted, output_path):
    fig, (ax, resax) = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    x, y, pred = fitted['x'], fitted['y'], fitted['prediction']

    ax.step(x, y, where='mid', color='.35', label='Observed')
    ax.plot(x, pred, color='midnightblue', label='Fitted model')

    ax.set(ylabel='Counts/channel', title=f"{spectrum['path'].name}: {fitted['name']}")

    ax.legend(frameon=False)

    r = poisson_deviance_residuals(y, pred)
    resax.plot(x, r, '.', color='black')
    resax.axhline(0, color='.5', ls='--')
    resax.set(xlabel='MCA channel', ylabel='Standardized residuals')

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def serializable_fit(fitted):
    return {k: v for k, v in fitted.items() if k not in
            {'x', 'y', 'prediction', 'params', 'center', 'slope_guess', 'background_order'}}

def fit_calibration_line(feature_rows):
    """
    Weighted line fit to N(E) = alpha*E + beta, inverted to get E = a*N + b
    """
    if len(feature_rows) < 2:
        raise ValueError('At least TWO independently identified features needed')
    
    ch = np.array([r['mu_ch'] for r in feature_rows], dtype=float)
    E = np.array([r['energy_keV'] for r in feature_rows], dtype=float)
    sig = np.array([r['mu_total_ch'] for r in feature_rows], dtype=float)

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
                chi2=chi2, dof=len(ch)-2, 
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

            sys_ch = max([abs(v) for v in shift] + [0])
            sys_ch = np.hypot(sys_ch, float(f.get('systematic_centroid_ch',0)))

            row = serializable_fit(base)

            row['mu_window_sys_ch'] = float(sys_ch)
            row['mu_total_ch'] = float(np.hypot(base['mu_stat_ch'], sys_ch))
            row['window_shifts_ch'] = shift

            rows.append(row)

            diagnostic_plot(s, base, out/f"feature_{f['name']}.png")

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

            ax.errorbar(ch, en, xerr=er, fmt='o', capsize=2, color="midnightblue", ecolor="midnightblue")
            xv = np.linspace(min(ch)-10, max(ch)+10, 200)

            ax.plot(xv, line['slope_keV_per_channel']*xv+line['offset_keV'])

            ax.set(xlabel='MCA channel', ylabel='Deposited energy (keV)', title='Overnight Ba-133 calibration')

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
    peak53_ref = next((f for f in features if f['kind'] == 'gaussian'
                       and np.isclose(f['energy_keV'], 53.1625, atol=0.1)), None)

    if cs_ref is None or ba_ref is None:
        raise ValueError('Overnight reference must contain the Cs K and 81 keV fits.')

    if cs_ref.get('beta_scale') is None:
        raise ValueError('Cs K fit needs an overnight K-beta amplitude scale.')

    scale = getattr(args, 'channel_scale', 1.0)
    shift = getattr(args, 'channel_shift', 0.0)

    if scale <= 0:
        raise ValueError('--channel-scale must be positive.')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(set(glob.glob(args.files, recursive=True)))
    if not paths:
        raise FileNotFoundError(f'No files matched {args.files}')

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

        row = {
            'file': str(path),
            'live_time_s': spectrum['live_time_s'],
            'method': 'daily_independent_multipoint',
        }

        try:
            slope = reference_slope / scale
            ba_first = fit_daily_peak(spectrum, ba_ref, slope)
            anchor_shift = ba_first['mu_ch'] - (scale * ba_ref['mu_ch'] + shift)

            # update the line separations as the daily slope is refined
            for iteration in range(2):
                fitted = []

                for ref in [cs_ref, ba_ref] + ([peak53_ref] if getattr(args, 'include_53', False)
                                                and peak53_ref is not None else []):
                    try:
                        fit = fit_daily_peak(spectrum, ref, slope, anchor_shift)
                        
                        # compare two nearby fit windows
                        alt = fit_daily_peak(spectrum, ref, slope, anchor_shift, shrink=2)

                        window_unc = abs(fit['mu_ch'] - alt['mu_ch'])
                        model_unc = scale * ref.get('mu_window_sys_ch', 0.0)

                        fit['mu_window_sys_ch'] = window_unc
                        fit['mu_total_ch'] = float(np.sqrt(fit['mu_stat_ch']**2 +
                                                          window_unc**2 + model_unc**2))

                        width_ratio = fit['width_ch'] / (scale * ref['width_ch'])

                        # reject unconstrained fits before calibrating
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

                line = fit_calibration_line(usable)
                slope = line['slope_keV_per_channel']

            for fit in fitted:
                entry = serializable_fit(fit)
                entry.update(file=str(path), use_in_calibration=fit['use_in_calibration'])
                feature_rows.append(entry)

                diagnostic_plot(spectrum, fit, out / f'{name}_{fit["name"]}.png')

            if len(usable) < 2:
                row.update(status='INSUFFICIENT_FEATURES', n_features=len(usable))
            else:
                row.update(line)
                row.update(status='PROVISIONAL',
                           gain_ratio_vs_overnight=reference_slope / slope,
                           used_features='; '.join(f['name'] for f in usable))

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
    t.add_argument('--files', required=True, help='Glob string, e.g. data/raw/calibration/*/*.Spe')
    t.add_argument('--output', required=True)
    t.add_argument('--include-53', action='store_true')
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