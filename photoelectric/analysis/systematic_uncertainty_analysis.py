import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from scipy.optimize import curve_fit
from scipy.constants import c,e

from method1_dubridge import fit_dubridge_per_run

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")

output_dir = Path("/home/rck67/jlab/photoelectric/data")
scenario_output = output_dir / "systematic_scenarios.csv"
summary_output = output_dir / "systematic_uncertainties.csv"

wavelengths = [546.1, 435.7, 365.0]

final_windows = {
    546.1: (4.7, 5.5),
    435.7: (5.1, 6.0),
    365.0: (5.5, 6.5),
}

# alternative fitting windows
window_options = {
    546.1: [
        (4.5, 5.5),
        (4.6, 5.5),
        (4.7, 5.5),
        (4.5, 6.0),
        (4.6, 6.0),
        (4.7, 6.0),
    ],

    435.7: [
        (5.0, 6.0),
        (5.1, 6.0),
        (5.2, 6.0),
        (5.0, 6.5),
        (5.1, 6.5),
        (5.2, 6.5),
    ],

    365.0: [
        (5.4, 6.5),
        (5.5, 6.5), 
        (5.6, 6.5),
        (5.4, 7.0),
        (5.5, 7.0), 
        (5.6, 7.0),
    ],
}

# wavelength filter uncertainty
wavelength_uncertainty_nm = 2.0

# voltmeter uncertainty
voltage_uncertainty_V = 0.01

# electrometer drift for each wavelength
drift_bias_nA = {
    546.1: 0.000,
    435.7: -0.001,
    365.0: -0.002
}

# electrometer resolution
current_resolution_nA = 0.001
sigma_resolution_nA = current_resolution_nA / np.sqrt(12)

def prepare_data(data):
    per_run = (
        data
        .groupby(
            ["Wavelength_nm", "Run", "Voltage_V"],
            as_index=False
        )
        .agg(
            Current_nA=("Current_nA", "mean")
        )
    )

    current_stats = (
        per_run
        .groupby(
            ["Wavelength_nm", "Voltage_V"],
            as_index=False
        )
        .agg(
            N_runs=("Current_nA", "count"),
            std_I_nA=("Current_nA", "std"),
        )
    )

    current_stats["sigma_I_nA"] = np.maximum(
        current_stats["std_I_nA"].fillna(0.0),
        sigma_resolution_nA
    )

    per_run = per_run.merge(
        current_stats[
            ["Wavelength_nm", "Voltage_V", "sigma_I_nA"]
        ],
        on=["Wavelength_nm", "Voltage_V"],
        how="left"
    )

    return per_run

# calculate cutoff voltage using different possible windows
def calculate_cutoffs(raw_data, windows):
    """
    Refit every run using the specified fitting window for each wavelength.
    """

    data = prepare_data(raw_data)

    rows = []
    for wavelength in wavelengths:

        vmin, vmax = windows[wavelength]
        wavelength_data = data[data["Wavelength_nm"] == wavelength]

        for run in sorted(wavelength_data["Run"].unique()):
            run_data = wavelength_data[wavelength_data["Run"] == run]

            result = fit_dubridge_per_run(run_data, vmin, vmax)

            rows.append({
                "Wavelength_nm": wavelength,
                "Run": run,
                "Vc_V": result["Vc"],
                "Vc_fit_err_V": result["Vc_err"],
            })

    run_results = pd.DataFrame(rows)

    summary_rows = []
    for wavelength in wavelengths:
        d = run_results[run_results["Wavelength_nm"] == wavelength]

        n = len(d)

        summary_rows.append({
            "Wavelength_nm": wavelength,
            "N_runs": n,
            "mean_Vc_V": d["Vc_V"].mean(),
            "std_Vc_V": d["Vc_V"].std(ddof=1),
            "SEM_Vc_V": d["Vc_V"].std(ddof=1) / np.sqrt(n),
        })

    cutoff_summary = pd.DataFrame(summary_rows)

    return run_results, cutoff_summary

# fit h
def linear(x, m, b):
    model = m*x + b
    return model

def fit_planck(cutoff_summary, wavelength_values=None, cutoff_offsets=None):
    data = cutoff_summary.copy()

    # default: using regular wavelengths w/o uncertainty
    if wavelength_values is None:
        lambda_nm = data["Wavelength_nm"].to_numpy(dtype=float)

    else:
        lambda_nm = np.array([
            wavelength_values[w]
            for w in data["Wavelength_nm"]
        ])

    nu_Hz = c / (lambda_nm * 1e-9)

    Vc = data["mean_Vc_V"].to_numpy(dtype=float)

    # voltage axis offsets
    if cutoff_offsets is not None:
        offsets = np.array([
            cutoff_offsets.get(w, 0.0)
            for w in data["Wavelength_nm"]
        ])

        Vc = Vc + offsets

    Vc_err = data["SEM_Vc_V"].to_numpy(dtype=float)

    # scale frequency so it doesn't fit to just high frequency
    nu_scaled = nu_Hz / 1e14

    popt, pcov = curve_fit(
        linear,
        nu_scaled,
        Vc,
        sigma=Vc_err,
        absolute_sigma=True
    )

    m_scaled, b = popt
    m_scaled_err, b_err = np.sqrt(np.diag(pcov))

    m = m_scaled / 1e14
    m_err = m_scaled_err / 1e14

    h = e * m
    h_stat = e * m_err

    phi = -e * b
    phi_stat = e * b_err

    return {
        "h": h,
        "h_stat": h_stat,
        "phi": phi,
        "phi_stat": phi_stat,
        "slope": m,
        "intercept": b,
    }

# evaluate systematic uncertainty contribution from window choice
def evaluate_window_systematic(raw_data, h_nominal):
    rows = []

    option_lists = [window_options[w] for w in wavelengths]

    for combo in product(*option_lists):
        windows = dict(zip(wavelengths, combo))

        try:
            _, summary = calculate_cutoffs(
                raw_data,
                windows
            )

            result = fit_planck(summary)

        except (ValueError, RuntimeError):
            continue

        delta_h = result["h"] - h_nominal

        rows.append({
            "source": "fitting window",
            "h_Js": result["h"],
            "delta_h_Js": delta_h,

            "546.1_window": str(windows[546.1]),
            "435.7_window": str(windows[435.7]),
            "365.0_window": str(windows[365.0]),
        })

    scenarios = pd.DataFrame(rows)

    # return max uncertainty on h
    sigma = np.max(np.abs(scenarios["delta_h_Js"]))

    return sigma, scenarios

# evaluate systematic uncertainy contribution from wavelength uncertainties
def evaluate_wavelength_systematic(cutoff_summary, h_nominal):
    rows = []

    # test all 8 combinations of +/- 2 nm
    for signs in product([-1, 1], repeat=len(wavelengths)):
        perturbed = {
            wavelength:
                wavelength
                + sign * wavelength_uncertainty_nm

            for wavelength, sign
            in zip(wavelengths, signs)
        }

        result = fit_planck(cutoff_summary, wavelength_values=perturbed)
        delta_h = result["h"] - h_nominal

        rows.append({
            "source": "wavelength",
            "h_Js": result["h"],
            "delta_h_Js": delta_h,
            "546.1_nm": perturbed[546.1],
            "435.7_nm": perturbed[435.7],
            "365.0_nm": perturbed[365.0],
        })

    scenarios = pd.DataFrame(rows)

    # get maximum uncertainty from different combinations wavelength uncertainties
    sigma = np.max(np.abs(scenarios["delta_h_Js"]))

    return sigma, scenarios

def evaluate_voltage_systematic(cutoff_summary, h_nominal):
    """
    If all three wavelengths have the same voltage offset, then phi may change but not h.

    However, if there is a different voltage offset between wavelengths, there will be an effect on h.
    """
    rows = []

    for signs in product([-1, 1], repeat=len(wavelengths)):
        offsets = {
            wavelength:
                sign * voltage_uncertainty_V

            for wavelength, sign
            in zip(wavelengths, signs)
        }

        result = fit_planck(cutoff_summary, cutoff_offsets=offsets)

        delta_h = result["h"] - h_nominal

        rows.append({
            "source": "voltage display",
            "h_Js": result["h"],
            "delta_h_Js": delta_h,
            "546.1_offset_V": offsets[546.1],
            "435.7_offset_V": offsets[435.7],
            "365.0_offset_V": offsets[365.0],
        })

    scenarios = pd.DataFrame(rows)

    sigma = np.max(np.abs(scenarios["delta_h_Js"]))

    return sigma, scenarios

def evaluate_drift_systematic(
    raw_data,
    h_nominal
):
    """
    Correct estimated current drift and rerun calculation of h.
    If measured current has bias of -x nA, then correction is +x nA.
    """
    corrected = raw_data.copy()

    for wavelength, bias in drift_bias_nA.items():
        mask = (
            corrected["Wavelength_nm"]
            == wavelength
        )

        corrected.loc[
            mask,
            "Current_nA"
        ] -= bias

    _, corrected_summary = calculate_cutoffs(corrected, final_windows)
    result = fit_planck(corrected_summary)

    delta_h = result["h"] - h_nominal

    scenarios = pd.DataFrame([{
        "source": "electrometer drift",
        "h_Js": result["h"],
        "delta_h_Js": delta_h,
    }])

    sigma = abs(delta_h)

    return sigma, scenarios

def main():
    raw_data = pd.read_csv(data_file)

    # nominal h and phi
    _, nominal_cutoffs = calculate_cutoffs(raw_data, final_windows)
    nominal_result = fit_planck(nominal_cutoffs)

    h_nominal = nominal_result["h"]
    h_stat = nominal_result["h_stat"]

    print(f"h = {h_nominal:.8e} +/- {h_stat:.8e} Js")

    # evaluate each systematic uncertainty individually
    sigma_window, window_scenarios = (
        evaluate_window_systematic(
            raw_data,
            h_nominal
        )
    )

    sigma_wavelength, wavelength_scenarios = (
        evaluate_wavelength_systematic(
            nominal_cutoffs,
            h_nominal
        )
    )

    sigma_voltage, voltage_scenarios = (
        evaluate_voltage_systematic(
            nominal_cutoffs,
            h_nominal
        )
    )

    sigma_drift, drift_scenarios = (
        evaluate_drift_systematic(
            raw_data,
            h_nominal
        )
    )

    # combine systematic uncertainties
    experimental_components = {
        "Wavelength filters": sigma_wavelength,
        "Voltmeter display": sigma_voltage,
        "Electrometer drift": sigma_drift,
    }

    sigma_total = np.sqrt(
        sum(
            value**2
            for value in experimental_components.values()
        )
    )

    print("\nSYSTEMATIC UNCERTAINTIES")
    print("=" * 70)

    for name, value in experimental_components.items():
        print(
            f"{name:25s}: "
            f"{value:.4e} Js "
            f"({100 * value / abs(h_nominal):.3f} %)"
        )

    print(f"\ntotal systematic uncertainty: {sigma_total:.4e} Js")

    # save all data
    all_scenarios = pd.concat(
        [
            window_scenarios,
            wavelength_scenarios,
            voltage_scenarios,
            drift_scenarios,
        ],
        ignore_index=True
    )

    all_scenarios.to_csv(
        scenario_output,
        index=False
    )

    summary_rows = [
        {
            "Source": name,
            "sigma_h_Js": value,
            "fraction percent":
                100 * value / abs(h_nominal),
        }

        for name, value
        in experimental_components.items()
    ]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_output, index=False)

    print(f"\nSaved scenarios to: {scenario_output}")

    print(f"Saved summary to: {summary_output}")

if __name__ == "__main__":
    main()