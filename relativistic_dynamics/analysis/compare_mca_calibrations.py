"""Compare daily MCA energy calibrations with the overnight reference.

Run as such:
python analysis/compare_mca_calibrations.py \
    --reference data/processed/calibration_results/overnight/overnight_calibration.json \
    --daily 'data/processed/calibration_results/daily/*/daily_calibrations.csv' \
    --output data/processed/calibration_results/comparison \
    --channel-min 40 --channel-max 750 --check-channels 55 145

Uncertainty bands neglect correlations between overnight and daily fit errors.
The daily two-point line cannot test linearity; high-channel comparisons extrapolate it.
"""

import argparse
import ast
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

COLORS = ["#B184C3", "#CA7EB1"]

def load_covariance(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError('Calibration covariance must be a finite 2x2 matrix.')
    return covariance


def variance_at_channel(channels, covariance):
    channels = np.asarray(channels, dtype=float)
    return np.maximum(covariance[0, 0] * channels**2
                      + 2 * covariance[0, 1] * channels
                      + covariance[1, 1], 0.)


def calibration_label(path):
    name = Path(path).stem
    date_string = name[:8]
    if len(date_string) == 8 and date_string.isdigit():
        return f'{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}'
    return name


def fit_fixed_slope_offset(features, slope, slope_var):
    """Fit the daily intercept with the overnight slope held fixed."""
    channels = features['mu_ch'].to_numpy(dtype=float)
    energies = features['energy_keV'].to_numpy(dtype=float)
    uncertainties = features['mu_total_ch'].to_numpy(dtype=float)

    if len(channels) != 2 or np.any(~np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
        raise ValueError('Fixed-slope calibration requires two valid centroid uncertainties.')

    weights = 1 / (slope * uncertainties)**2
    offset_estimates = energies - slope * channels
    offset = float(np.average(offset_estimates, weights=weights))
    offset_err_conditional = float(np.sqrt(1 / np.sum(weights)))
    mean_channel = float(np.average(channels, weights=weights))

    residuals = offset_estimates - offset
    chi2_conditional = float(np.sum(weights * residuals**2))

    # Include uncertainty in the transferred slope for the one-dof consistency test.
    span = channels[1] - channels[0]
    delta_residual = offset_estimates[1] - offset_estimates[0]
    variance_span = slope**2 * np.sum(uncertainties**2) + span**2 * slope_var
    chi2_with_slope_unc = float(delta_residual**2 / variance_span)

    return {
        'fixed_slope_keV_per_channel': float(slope),
        'fixed_slope_offset_keV': offset,
        'fixed_slope_offset_conditional_err_keV': offset_err_conditional,
        'fixed_slope_mean_channel': mean_channel,
        'fixed_slope_chi2_conditional': chi2_conditional,
        'fixed_slope_chi2_with_ref_slope_unc': chi2_with_slope_unc,
        'fixed_slope_dof': 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--daily', required=True,
                        help='Daily CSV pathname or quoted glob of daily CSVs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--channel-min', type=int, default=40)
    parser.add_argument('--channel-max', type=int, default=170)
    parser.add_argument('--check-channels', nargs='+', type=int, default=None,
                        help='Channels reported separately; defaults to reference Cs and 81-keV anchors')
    args = parser.parse_args()

    if not (0 <= args.channel_min < args.channel_max <= 2047):
        parser.error('Require 0 <= --channel-min < --channel-max <= 2047')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference_json = json.loads(Path(args.reference).read_text())
    reference = reference_json['linear_calibration']

    cs_ref = next((f for f in reference_json['features']
                   if f['kind'] == 'cs_k_complex'), None)
    ba_ref = next((f for f in reference_json['features']
                   if f['kind'] == 'doublet'), None)
    if cs_ref is None or ba_ref is None:
        raise ValueError('Overnight JSON requires Cs K and 81 keV reference features.')
    if not np.isclose(cs_ref['energy_keV'], 30.97313, atol=0.001):
        raise ValueError('Overnight Cs Kalpha1 calibration point must have E = 30.97313 keV.')

    if args.check_channels is None:
        args.check_channels = [int(round(cs_ref['mu_ch'])),
                               int(round(ba_ref['mu_ch']))]
    if any(not 0 <= n <= 2047 for n in args.check_channels):
        parser.error('All check channels must be between 0 and 2047')

    a_ref = float(reference['slope_keV_per_channel'])
    b_ref = float(reference['offset_keV'])
    cov_ref = load_covariance(reference['covariance_keV'])

    csv_paths = sorted(glob.glob(args.daily, recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f'No daily calibration CSV matches {args.daily}')

    daily = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    daily = daily[daily['status'].eq('PROVISIONAL')].copy()
    daily = daily[daily['n_features'].eq(2)].copy()
    daily = daily.drop_duplicates(subset=['date'], keep='last')
    if daily.empty:
        raise ValueError('No usable two-feature daily calibrations found.')

    feature_tables = []
    for csv_path in csv_paths:
        feature_path = Path(csv_path).with_name('daily_features.csv')
        if not feature_path.exists():
            raise FileNotFoundError(f'Fixed-slope fit requires {feature_path}')
        feature_tables.append(pd.read_csv(feature_path))

    daily_features = pd.concat(feature_tables, ignore_index=True)
    daily_features = daily_features[
        daily_features['use_in_calibration'].astype(str).str.lower().eq('true')
    ].copy()

    plot_ch = np.arange(args.channel_min, args.channel_max + 1)
    full_ch = np.arange(2048, dtype=int)
    summary_rows, difference_rows = [], []
    fixed_rows, fixed_difference_rows = [], []
    anchor_residual_rows = []

    fig_fixed, ax_fixed = plt.subplots(figsize=(8, 4.7))
    ax_fixed.axhline(0, color='0.55', ls='--', lw=1)

    fig_diff, ax_diff = plt.subplots(figsize=(8, 4.7))
    fig_overlay, ax_overlay = plt.subplots(figsize=(8, 4.7))
    ax_diff.axhline(0, color='0.55', ls='--', lw=1)
    ax_overlay.plot(plot_ch, a_ref * plot_ch + b_ref,
                    color="midnightblue", lw=1.5, label='Overnight reference')

    for i, (_, run) in enumerate(daily.sort_values('file').iterrows()):
        a = float(run['slope_keV_per_channel'])
        b = float(run['offset_keV'])
        cov = load_covariance(run['covariance_keV'])
        da, db = a - a_ref, b - b_ref
        fractional = da / a_ref

        # Approximate difference covariance assumes independent fitted coefficients.
        covariance_difference = cov + cov_ref
        da_err = np.sqrt(covariance_difference[0, 0])
        db_err = np.sqrt(covariance_difference[1, 1])
        frac_err = np.sqrt(cov[0, 0] / a_ref**2
                           + a**2 * cov_ref[0, 0] / a_ref**4)

        label = str(run['date']) if pd.notna(run.get('date')) else calibration_label(run['file'])
        color = COLORS[i % len(COLORS)]
        dE = da * plot_ch + db
        dE_err = np.sqrt(variance_at_channel(plot_ch, covariance_difference))

        ax_diff.plot(plot_ch, dE, color=color, lw=1.5, label=label)
        ax_diff.fill_between(plot_ch, dE - dE_err, dE + dE_err,
                             color=color, alpha=0.14)
        ax_overlay.plot(plot_ch, a * plot_ch + b, color=color,
                        lw=1.0, ls='--', label=label)

        summary = {
            'file': str(run['file']),
            'date': label,
            'daily_slope_keV_per_channel': a,
            'daily_offset_keV': b,
            'delta_slope_keV_per_channel': da,
            'delta_slope_approx_err': da_err,
            'delta_offset_keV': db,
            'delta_offset_approx_err_keV': db_err,
            'fractional_slope_change': fractional,
            'fractional_slope_change_approx_err': frac_err,
            'fractional_slope_change_percent': 100 * fractional,
        }

        full_dE = da * full_ch + db
        full_err = np.sqrt(variance_at_channel(full_ch, covariance_difference))
        for n, value, uncertainty in zip(full_ch, full_dE, full_err):
            difference_rows.append({'date': label, 'file': str(run['file']),
                                    'channel': int(n), 'delta_energy_keV': value,
                                    'approx_delta_energy_err_keV': uncertainty})

        for n in args.check_channels:
            summary[f'delta_E_ch{n}_keV'] = da * n + db
            summary[f'delta_E_ch{n}_approx_err_keV'] = float(np.sqrt(
                variance_at_channel(n, covariance_difference)))

        summary_rows.append(summary)

        anchors = daily_features[daily_features['file'].eq(str(run['file']))]
        anchors = anchors[anchors['kind'].isin(['cs_k_complex', 'doublet'])]
        if set(anchors['kind']) != {'cs_k_complex', 'doublet'} or len(anchors) != 2:
            raise ValueError(f'{label}: need one accepted Cs K and one accepted 81 keV fit in daily_features.csv')
        anchors = anchors.sort_values('energy_keV')

        fixed = fit_fixed_slope_offset(anchors, a_ref, cov_ref[0, 0])
        b_fixed = fixed['fixed_slope_offset_keV']
        mean_ch = fixed['fixed_slope_mean_channel']

        # Daily centroid errors and the overnight slope/offset covariance.
        reference_gradient = np.array([-mean_ch, -1.0])
        delta_b_variance = (fixed['fixed_slope_offset_conditional_err_keV']**2
                            + reference_gradient @ cov_ref @ reference_gradient)
        delta_b_err = float(np.sqrt(max(delta_b_variance, 0.0)))
        delta_b_fixed = b_fixed - b_ref

        slope_var = cov_ref[0, 0]
        conditional_var = fixed['fixed_slope_offset_conditional_err_keV']**2
        fixed_cov = np.array([
            [slope_var, -mean_ch * slope_var],
            [-mean_ch * slope_var, conditional_var + mean_ch**2 * slope_var],
        ])

        fixed_row = dict(date=label, file=str(run['file']), **fixed,
                         fixed_slope_offset_approx_err_keV=float(np.sqrt(fixed_cov[1, 1])),
                         fixed_slope_covariance_keV=fixed_cov.tolist(),
                         fixed_minus_overnight_offset_keV=delta_b_fixed,
                         fixed_minus_overnight_offset_approx_err_keV=delta_b_err,
                         free_slope_keV_per_channel=a,
                         free_offset_keV=b)

        for n in args.check_channels:
            fixed_row[f'fixed_minus_free_ch{n}_keV'] = (a_ref - a) * n + b_fixed - b

        dof = fixed['fixed_slope_dof']
        fit_chi2 = fixed['fixed_slope_chi2_conditional']
        fit_chi2_with_slope_unc = fixed['fixed_slope_chi2_with_ref_slope_unc']

        fixed_row['fixed_slope_reduced_chi2'] = fit_chi2 / dof
        fixed_row['fixed_slope_p_value'] = float(chi2.sf(fit_chi2, dof))
        fixed_row['fixed_slope_p_with_ref_slope_unc'] = float(
            chi2.sf(fit_chi2_with_slope_unc, dof))

        print(f'Including overnight slope uncertainty: '
              f'chi2/dof = {fit_chi2_with_slope_unc:.3f}/{dof}')
        print('  Reference            Measured ch     Predicted ch     Residual ch')

        for _, anchor in anchors.iterrows():
            observed_ch = float(anchor['mu_ch'])
            channel_unc = float(anchor['mu_total_ch'])
            predicted_ch = (float(anchor['energy_keV']) - b_fixed) / a_ref
            residual_ch = observed_ch - predicted_ch

            anchor_residual_rows.append({
                'date': label,
                'feature': anchor['name'],
                'energy_keV': float(anchor['energy_keV']),
                'measured_channel': observed_ch,
                'channel_uncertainty': channel_unc,
                'predicted_channel': predicted_ch,
                'residual_channel': residual_ch,
                'residual_over_channel_uncertainty': residual_ch / channel_unc,
            })

            print(f'  {anchor["name"]:<18} {observed_ch:>10.3f} '
                  f'{predicted_ch:>16.3f} {residual_ch:>15.3f}')

        fixed_rows.append(fixed_row)

        fixed_change = np.full_like(plot_ch, delta_b_fixed, dtype=float)
        ax_fixed.plot(plot_ch, dE, color=color, lw=1.0, ls=':',
                      label=f'{label} (free slope)')
        ax_fixed.plot(plot_ch, fixed_change, color=color, lw=1.5,
                      label=f'{label} (fixed slope)')
        ax_fixed.fill_between(plot_ch, delta_b_fixed - delta_b_err,
                              delta_b_fixed + delta_b_err, color=color, alpha=0.14)

        for n in full_ch:
            fixed_difference_rows.append({
                'date': label,
                'file': str(run['file']),
                'channel': int(n),
                'overnight_energy_keV': a_ref * n + b_ref,
                'free_slope_daily_energy_keV': a * n + b,
                'fixed_slope_daily_energy_keV': a_ref * n + b_fixed,
                'fixed_slope_daily_energy_approx_err_keV': float(np.sqrt(
                    variance_at_channel(n, fixed_cov))),
                'fixed_minus_overnight_keV': delta_b_fixed,
                'fixed_minus_overnight_approx_err_keV': delta_b_err,
                'fixed_minus_free_keV': (a_ref - a) * n + b_fixed - b,
            })

    ax_diff.set(xlabel='MCA channel', ylabel=r'$\Delta E(n) = E_{daily} - E_{ref}$ (keV)',
                title='Daily MCA calibration shifts relative to overnight')
    ax_diff.legend(frameon=False, fontsize=8)
    fig_diff.tight_layout()
    fig_diff.savefig(out/'energy_difference.png', dpi=200)
    plt.close(fig_diff)

    ax_overlay.set(xlabel='MCA channel', ylabel='Deposited energy (keV)',
                   title='Daily and overnight energy calibrations')
    ax_overlay.legend(frameon=False, fontsize=8)
    fig_overlay.tight_layout()
    fig_overlay.savefig(out/'calibration_overlay.png', dpi=200)
    plt.close(fig_overlay)

    ax_fixed.set(xlabel='MCA channel',
                 ylabel=r'$E_{daily} - E_{ref}$ (keV)',
                 title='Fixed-slope and free-slope daily calibration shifts')
    ax_fixed.legend(frameon=False, fontsize=8)
    fig_fixed.tight_layout()
    fig_fixed.savefig(out/'fixed_slope_energy_difference.png', dpi=200)
    plt.close(fig_fixed)

    pd.DataFrame(fixed_rows).to_csv(out/'fixed_slope_comparison.csv', index=False)
    pd.DataFrame(anchor_residual_rows).to_csv(
        out/'fixed_slope_anchor_residuals.csv', index=False)
    pd.DataFrame(fixed_difference_rows).to_csv(
        out/'fixed_slope_energy_difference_by_channel.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out/'calibration_comparison.csv', index=False)
    pd.DataFrame(difference_rows).to_csv(out/'energy_difference_by_channel.csv', index=False)

    with pd.option_context('display.float_format', '{:.5g}'.format,
                           'display.width', 160):
        print(summary_df[['date', 'delta_slope_keV_per_channel', 'delta_offset_keV',
                          'fractional_slope_change_percent']].to_string(index=False))
    print(f'\nSaved comparison results to {out}')

if __name__ == '__main__':
    main()
