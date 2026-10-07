import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from pathlib import Path
from method2_piecewise import prepare_run_data, fit_piecewise_per_run

sns.set_theme(style="ticks", context="paper")

data_file = Path("/home/rck67/jlab/photoelectric/data/photoelectric_data_clean.csv")
detailed_output_file = Path("/home/rck67/jlab/photoelectric/data/method2_window_study_by_run.csv")
summary_output_file = Path("/home/rck67/jlab/photoelectric/data/method2_window_study_summary.csv")
figure_dir = Path("/home/rck67/jlab/photoelectric/figures/method2_window_study")
figure_dir.mkdir(parents=True, exist_ok=True)

wavelengths = [546.1, 435.7, 365.0]

window_search = {
    546.1: {
        "vmin_range": (3.5, 4.7),
        "vmax_range": (5.5, 7.0),
    },
    435.7: {
        "vmin_range": (4.0, 5.2),
        "vmax_range": (6.0, 7.5),
    },
    365.0: {
        "vmin_range": (4.5, 5.7),
        "vmax_range": (6.5, 8.0),
    },
}

def get_bin_edges(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 1:
        return np.array([values[0] - 0.05, values[0] + 0.05])

    edges = np.empty(len(values) + 1)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - (values[1] - values[0]) / 2
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2
    return edges

def candidate_bounds(wavelength_data, low, high):
    voltages = np.sort(wavelength_data["Voltage_V"].unique())
    return voltages[(voltages >= low) & (voltages <= high)]

def plot_numeric_heatmap(d, value_column, wavelength, title, filename, fmt):
    pivot = (
        d.pivot(index="Vmax_V", columns="Vmin_V", values=value_column)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    vmins = pivot.columns.to_numpy(dtype=float)
    vmaxs = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)

    x_edges = get_bin_edges(vmins)
    y_edges = get_bin_edges(vmaxs)

    fig, ax = plt.subplots(figsize=(10, 7))

    norm = mcolors.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    cmap = sns.color_palette("mako_r", as_cmap=True)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=cmap,
        shading="flat",
    )

    for i, vmax in enumerate(vmaxs):
        for j, vmin in enumerate(vmins):
            value = z[i, j]

            if np.isfinite(value):
                rgba = cmap(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"

                ax.text(
                    vmin,
                    vmax,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    ax.set_xticks(vmins)
    ax.set_yticks(vmaxs)
    ax.set_xlabel(r"Lower fit bound $V_{\min}$ (V)")
    ax.set_ylabel(r"Upper fit bound $V_{\max}$ (V)")
    ax.set_title(title)

    sns.despine()
    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    data = prepare_run_data(data_file)
    detailed_rows = []

    for wavelength in wavelengths:
        wavelength_data = data[data["Wavelength_nm"] == wavelength].copy()
        runs = sorted(wavelength_data["Run"].unique())

        settings = window_search[wavelength]
        vmins = candidate_bounds(wavelength_data, *settings["vmin_range"])
        vmaxs = candidate_bounds(wavelength_data, *settings["vmax_range"])

        for vmin in vmins:
            for vmax in vmaxs:
                if vmax <= vmin:
                    continue

                for run in runs:
                    run_data = wavelength_data[wavelength_data["Run"] == run]

                    try:
                        result = fit_piecewise_per_run(run_data, vmin, vmax)
                    except (ValueError, RuntimeError):
                        continue

                    detailed_rows.append({
                        "Wavelength_nm": wavelength,
                        "Run": run,
                        "Vmin_V": vmin,
                        "Vmax_V": vmax,
                        "N_points": result["N_points"],
                        "Vc_V": result["Vc"],
                        "Vc_fit_err_V": result["Vc_err"],
                        "chi2": result["chi2"],
                        "dof": result["dof"],
                        "reduced_chi2": result["reduced_chi2"],
                    })

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(detailed_output_file, index=False)

    summary = (
        detailed
        .groupby(["Wavelength_nm", "Vmin_V", "Vmax_V"], as_index=False)
        .agg(
            N_successful_runs=("Run", "count"),
            mean_Vc_V=("Vc_V", "mean"),
            std_Vc_V=("Vc_V", "std"),
            mean_Vc_fit_err_V=("Vc_fit_err_V", "mean"),
            mean_chi2=("chi2", "mean"),
            mean_reduced_chi2=("reduced_chi2", "mean"),
            max_reduced_chi2=("reduced_chi2", "max"),
            total_chi2=("chi2", "sum"),
            total_dof=("dof", "sum"),
        )
    )

    summary["combined_reduced_chi2"] = summary["total_chi2"] / summary["total_dof"]
    summary["all_runs_successful"] = summary["N_successful_runs"] == 5
    summary.to_csv(summary_output_file, index=False)

    complete = summary[summary["all_runs_successful"]].copy()

    for wavelength in wavelengths:
        d = complete[complete["Wavelength_nm"] == wavelength]

        if d.empty:
            print(f"No complete Method 2 windows for {wavelength} nm")
            continue

        plot_numeric_heatmap(
            d,
            value_column="mean_Vc_V",
            wavelength=wavelength,
            title=fr"Method 2: mean $V_c$, {wavelength} nm",
            filename=f"mean_Vc_{wavelength}nm.png",
            fmt=".4f"
        )

        plot_numeric_heatmap(
            d,
            value_column="std_Vc_V",
            wavelength=wavelength,
            title=fr"Method 2: run-to-run standard deviation of $V_c$, {wavelength} nm",
            filename=f"Vc_scatter_{wavelength}nm.png",
            fmt=".4f"
        )

        plot_numeric_heatmap(
            d,
            value_column="mean_reduced_chi2",
            wavelength=wavelength,
            title=fr"Method 2: mean $\chi^2/\mathrm{{dof}}$, {wavelength} nm",
            filename=f"reduced_chi2_{wavelength}nm.png",
            fmt=".2f"
        )

    print(f"Saved Method 2 individual-run window fits to: {detailed_output_file}")
    print(f"Saved Method 2 window summaries to: {summary_output_file}")
    print(f"Saved Method 2 window-study figures to: {figure_dir}")

if __name__ == "__main__":
    main()
