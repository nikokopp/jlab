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

def compton_edge_keV(gamma_energy_keV):
    """Maximum electron kinetic energy for 180-degree gamma scattering."""
    return gamma_energy_keV * (1 - 1 / (1 + 2 * gamma_energy_keV / 510.99895))

def model(x, params, kind, center, slope_guess, fixed_width=None, background_order=1):
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
        feature = amp * np.exp(-0.5 * ((x - mu) / width)**2)
    elif kind == 'doublet':
        delta = (E_81 - E_79) / slope_guess
        feature = amp * (np.exp(-0.5 * ((x - mu) / width)**2) + R_79_81 * np.exp(-0.5 * ((x - (mu - delta)) / width)**2))
    elif kind == 'edge':
        feature = 0.5 * amp * erfc((x - mu) / (np.sqrt(2) * width))
    else:
        raise ValueError(f'Unknown feature kind: {kind}')
    return feature + background

def poisson_deviance_residuals(y, prediction):
    prediction = np.maximum(prediction, 1e-10)
    dev = 2 * (prediction - y + xlogy(y, y / prediction))
    return np.sign(y - prediction) * np.sqrt(np.maximum(dev, 0))

def fit_feature(spectrum, feature, slope_guess, *, fixed_width=None):
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
        lo += [-.04]
        hi += [.04]
    init, lo, hi = np.asarray(init), np.asarray(lo), np.asarray(hi)
    init = np.maximum(lo + 1e-5, np.minimum(hi - 1e-5, init))


    def predict(p, xv=x):
        return model(xv, p, kind, center, slope_guess, fixed_width, order)

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
    
    return dict(name=feature['name'], 
                kind=kind, 
                energy_keV=float(feature['energy_keV']),
                mu_ch=float(outcome.x[1]), 
                mu_stat_ch=mu_stat,
                width_ch=width, 
                chi2=chi2, 
                dof=dof,
                reduced_chi2=chi2/dof if dof > 0 else float('nan'),
                bound_hit=bool(bound_hit), 
                x=x, y=y, 
                prediction=pred,
                params=outcome.x, 
                center=center, 
                slope_guess=slope_guess,
                background_order=order
                )

def diagnostic_plot(spectrum, fitted, output_path):
    fig, (ax, resax) = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    x, y, pred = fitted['x'], fitted['y'], fitted['prediction']

    ax.step(x, y, where='mid', color='black', label='Original data')
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
    """Weighted n(E)=alpha*E+beta, inverted to E=a*n+b.

    The reference photon/edge energies are treated as exact, while the measured
    centroids have (approximately Gaussian) uncertainties in channel units.
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
                chi2=chi2, 
                dof=len(ch)-2, 
                reduced_chi2=chi2/dof,
                n_features=len(ch),
                residuals_keV=(E-(a*ch+b)).tolist()
                )

def inspect(spe_path, out):
    s = read_spe(spe_path)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.step(s['channels'], s['counts'], where='mid', lw=.9)

    ax.set(xlabel='MCA channel', ylabel='Counts/channel', 
           yscale='symlog',
           title=f"{s['path'].name} (live time {s['live_time_s']:.0f} s)")
    
    fig.tight_layout()

    fig.savefig(out/'overview.png', dpi=180)

    plt.close(fig)

    for hi in (250, 750):
        fig, ax = plt.subplots(figsize=(10, 5))
        mask = s['channels'] <= hi

        ax.step(s['channels'][mask], s['counts'][mask], where='mid', lw=.9)
        ax.set(xlabel='MCA channel', ylabel='Counts/channel', 
               yscale='symlog',
               title=f'{s["path"].name}: channels 0-{hi}')
        
        fig.tight_layout()
        fig.savefig(out/f'overview_0_to_{hi}.png', dpi=180)
        plt.close(fig)

    print('Saved whole-spectrum and two zoomed overviews; verify peak IDs first.')


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

            # largest observed sensitivity to window
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

    result = dict(reference_spectrum=str(s['path']), 
                  live_time_s=s['live_time_s'],
                  slope_guess=float(args.slope_guess), 
                  features=rows)
    
    if len(rows) >= 2:
        try:
            result['linear_calibration'] = fit_calibration_line(rows)
            print('Calibration E(keV) = a*channel + b:', result['linear_calibration'])

            line = result['linear_calibration']

            fig, ax = plt.subplots(figsize=(6.8, 4.5))
            ch = np.array([r['mu_ch'] for r in rows])
            en = np.array([r['energy_keV'] for r in rows])
            er = np.array([r['mu_total_ch'] for r in rows])

            ax.errorbar(ch, en, xerr=er, fmt='o', capsize=2)

            xv = np.linspace(min(ch)-10, max(ch)+10, 200)
            ax.plot(xv, line['slope_keV_per_channel']*xv+line['offset_keV'])

            ax.set(xlabel='MCA channel', 
                   ylabel='Deposited energy (keV)',
                   title='Overnight MCA energy calibration')
            
            fig.tight_layout()
            fig.savefig(out/'overnight_calibration_line.png', dpi=180)
            plt.close(fig)

        except ValueError as exc:
            warnings.warn(str(exc))
    else:
        print('Only one verified feature: no measured slope/offset can be calculated yet.')
    (out/'overnight_calibration.json').write_text(json.dumps(result, indent=2))


def run_daily(args):
    reference = json.loads(Path(args.reference).read_text())

    if 'linear_calibration' not in reference:
        raise ValueError('Overnight JSON has no independent multi-point calibration')
    
    cal = reference['linear_calibration']
    a = cal['slope_keV_per_channel']
    a_err = cal['slope_err_keV_per_channel']

    feature = next((f for f in reference['features'] if f['kind']=='doublet'
                    and np.isclose(f['energy_keV'], E_81, atol=.02)), None)
    
    if feature is None:
        raise ValueError('Reference needs a fitted Ba 81-keV doublet')
    
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(set(glob.glob(args.files, recursive=True)))

    if not paths:
        raise FileNotFoundError(f'No files matched {args.files}')
    
    rows = []
    for path in paths:
        s = read_spe(path)
        mu_ref = feature['mu_ch']

        # widen window slightly to allow daily drift
        w = max(13., 3.5*feature['width_ch'])

        conf = dict(name='81_keV_peak', 
                    kind='doublet', 
                    energy_keV=E_81,
                    expected_channel=mu_ref,
                    window=[max(0.,mu_ref-w), mu_ref+w], 
                    background_order=1)
        try:
            fit = fit_feature(s, conf, a, fixed_width=feature['width_ch'])

            daily_mu = fit['mu_ch']
            b = E_81 - a*daily_mu

            row = dict(file=str(path), live_time_s=s['live_time_s'],
                       method='81keV_anchor_slope_transferred',
                       slope_keV_per_channel=a,
                       slope_err_keV_per_channel=a_err,
                       offset_keV=b, anchor_channel=daily_mu,
                       anchor_stat_err_channel=fit['mu_stat_ch'],
                       anchor_shift_channel=daily_mu-mu_ref,
                       anchor_energy_keV=E_81,
                       reduced_deviance=fit['reduced_deviance'],
                       hit_parameter_bound=fit['bound_hit'])
            
            rows.append(row)

            diagnostic_plot(s, fit, out/f'{Path(path).stem}_81keV_daily.png')

            print(f"{Path(path).name}: anchor={daily_mu:.2f} ±"
                  f" {fit['mu_stat_ch']:.2f} channels; slope transferred")
            
        except (ValueError, RuntimeError) as exc:
            warnings.warn(f'Daily fit FAILED for {path}: {exc}')
            rows.append(dict(file=str(path), method='FAILED', problem=str(exc)))

    pd.DataFrame(rows).to_csv(out/'daily_calibrations.csv', index=False)
    print(f'Saved {out/"daily_calibrations.csv"}; inspect each plot and flags.')

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