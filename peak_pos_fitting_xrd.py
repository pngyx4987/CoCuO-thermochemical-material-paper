# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:03:45 2026

@author: pyx
"""

import os
import glob
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ============================================================
# USER INPUT
# ============================================================

run_numbers = [
    "185401",
    "185434",
    "185507",
    "185542",
    "185619",
]

base_dir = (
    r"C:/Users/pyx/Desktop/documents/SBU WORK/DEGREE/"
    r"XPD_CuO_Co3O4/refinement_CuCoO/"
    r"baseline_corrected - YXP"
)

# CoO peak fitting range
fit_min = 4.7
fit_max = 4.9

# Each scan takes 30 s
seconds_per_scan = 30

# Plot settings
figure_size = (8, 6)
marker_size = 55
line_width = 1.8

# Set True to inspect every individual peak fit
show_individual_fits = True

# Save final figure and fitted values
save_result = False
output_dpi = 300


# ============================================================
# FITTING FUNCTION
#
# Gaussian peak + linear background
# ============================================================

def gaussian_linear(x, amplitude, center, sigma, slope, intercept):

    gaussian = amplitude * np.exp(
        -0.5 * ((x - center) / sigma) ** 2
    )

    background = slope * x + intercept

    return gaussian + background


# ============================================================
# EXTRACT TIME FROM FILE NAME
#
# Expected filename pattern:
#
# YYYYMMDD-HHMMSS-N
#
# For example:
# 20241004-185619-9
#
# The scan index N adds N × 30 s.
# ============================================================

def extract_scan_time(file_path, seconds_per_scan=30):

    file_name = os.path.basename(file_path)

    match = re.search(
        r"(\d{8})-(\d{6})-(\d+)",
        file_name
    )

    if match is None:
        return None

    date_string = match.group(1)
    time_string = match.group(2)
    scan_index = int(match.group(3))

    base_time = datetime.strptime(
        date_string + time_string,
        "%Y%m%d%H%M%S"
    )

    scan_time = base_time + timedelta(
        seconds=scan_index * seconds_per_scan
    )

    return scan_time


# ============================================================
# FIND ALL REQUIRED FILES
#
# 185401, 185434, 185507, and 185542:
#     include scan index 0 only
#
# 185619:
#     include scan indices 0 through 9
# ============================================================

selected_files = []


for run in run_numbers:

    # --------------------------------------------------------
    # Batch run: 185619-0 through 185619-9
    # --------------------------------------------------------

    if run == "185619":

        for scan_index in range(10):

            pattern = os.path.join(
                base_dir,
                f"*{run}-{scan_index}*.xy"
            )

            matches = glob.glob(pattern)

            matches = [
                file_path
                for file_path in matches
                if "NA" not in os.path.basename(file_path)
            ]

            if len(matches) == 0:

                print(
                    f"No file found for "
                    f"{run}-{scan_index}"
                )

                continue

            matches = sorted(matches)

            if len(matches) > 1:

                print(
                    f"\nMultiple files found for "
                    f"{run}-{scan_index}:"
                )

                for file_path in matches:
                    print(
                        "   ",
                        os.path.basename(file_path)
                    )

            selected_file = matches[0]

            selected_files.append(
                {
                    "run": run,
                    "scan_index": scan_index,
                    "label": f"{run}-{scan_index}",
                    "file_path": selected_file,
                    "scan_time": extract_scan_time(
                        selected_file,
                        seconds_per_scan
                    )
                }
            )

            print(
                f"Selected {run}-{scan_index}: "
                f"{os.path.basename(selected_file)}"
            )

    # --------------------------------------------------------
    # Single scans: use index 0 only
    # --------------------------------------------------------

    else:

        pattern = os.path.join(
            base_dir,
            f"*{run}-0*.xy"
        )

        matches = glob.glob(pattern)

        matches = [
            file_path
            for file_path in matches
            if "NA" not in os.path.basename(file_path)
        ]

        # Fallback in case "-0" is not present in the filename
        if len(matches) == 0:

            pattern = os.path.join(
                base_dir,
                f"*{run}*.xy"
            )

            matches = glob.glob(pattern)

            matches = [
                file_path
                for file_path in matches
                if "NA" not in os.path.basename(file_path)
            ]

        if len(matches) == 0:

            print(
                f"No file found for run {run}"
            )

            continue

        matches = sorted(matches)

        if len(matches) > 1:

            print(
                f"\nMultiple files found for run {run}:"
            )

            for file_path in matches:
                print(
                    "   ",
                    os.path.basename(file_path)
                )

        selected_file = matches[0]

        selected_files.append(
            {
                "run": run,
                "scan_index": 0,
                "label": f"{run}-0",
                "file_path": selected_file,
                "scan_time": extract_scan_time(
                    selected_file,
                    seconds_per_scan
                )
            }
        )

        print(
            f"Selected {run}-0: "
            f"{os.path.basename(selected_file)}"
        )


if len(selected_files) == 0:

    raise FileNotFoundError(
        "No matching XY files were found."
    )


# ============================================================
# SORT FILES CHRONOLOGICALLY
# ============================================================

if all(
    item["scan_time"] is not None
    for item in selected_files
):

    selected_files.sort(
        key=lambda item: item["scan_time"]
    )

else:

    print(
        "\nWarning: the scan time could not be extracted "
        "from every filename."
    )


# ============================================================
# FIT EACH CoO PEAK
# ============================================================

results = []


for item in selected_files:

    run = item["run"]
    scan_index = item["scan_index"]
    scan_label = item["label"]
    file_path = item["file_path"]
    scan_time = item["scan_time"]

    # --------------------------------------------------------
    # Load XY data
    # --------------------------------------------------------

    data = np.loadtxt(file_path)

    data = np.atleast_2d(data)

    if data.shape[1] < 2:

        print(
            f"Skipping {scan_label}: "
            "fewer than two columns were found."
        )

        continue

    x_all = data[:, 0]
    y_all = data[:, 1]

    # --------------------------------------------------------
    # Select 4.7–4.9° fitting range
    # --------------------------------------------------------

    valid = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (x_all >= fit_min)
        & (x_all <= fit_max)
    )

    x = x_all[valid]
    y = y_all[valid]

    if len(x) < 7:

        print(
            f"Skipping {scan_label}: "
            "not enough points in the fitting range."
        )

        continue

    sort_order = np.argsort(x)

    x = x[sort_order]
    y = y[sort_order]

    # --------------------------------------------------------
    # Estimate initial linear background
    # --------------------------------------------------------

    n_edge = max(
        2,
        len(x) // 10
    )

    edge_x = np.concatenate(
        [
            x[:n_edge],
            x[-n_edge:]
        ]
    )

    edge_y = np.concatenate(
        [
            y[:n_edge],
            y[-n_edge:]
        ]
    )

    background_coefficients = np.polyfit(
        edge_x,
        edge_y,
        1
    )

    slope_guess = background_coefficients[0]
    intercept_guess = background_coefficients[1]

    background_guess = (
        slope_guess * x
        + intercept_guess
    )

    y_corrected = y - background_guess

    # --------------------------------------------------------
    # Initial peak parameters
    # --------------------------------------------------------

    peak_index = np.argmax(
        y_corrected
    )

    center_guess = x[peak_index]

    amplitude_guess = max(
        y_corrected[peak_index],
        np.ptp(y) * 0.5,
        1e-8
    )

    sigma_guess = 0.01

    initial_guess = [
        amplitude_guess,
        center_guess,
        sigma_guess,
        slope_guess,
        intercept_guess
    ]

    # --------------------------------------------------------
    # Fit bounds
    # --------------------------------------------------------

    lower_bounds = [
        0,
        fit_min,
        0.0005,
        -np.inf,
        -np.inf
    ]

    upper_bounds = [
        np.inf,
        fit_max,
        0.08,
        np.inf,
        np.inf
    ]

    # --------------------------------------------------------
    # Fit Gaussian + linear background
    # --------------------------------------------------------

    try:

        fitted_parameters, covariance = curve_fit(
            gaussian_linear,
            x,
            y,
            p0=initial_guess,
            bounds=(
                lower_bounds,
                upper_bounds
            ),
            maxfev=20000
        )

    except RuntimeError:

        print(
            f"Peak fitting failed for {scan_label}"
        )

        continue

    amplitude_fit = fitted_parameters[0]
    center_fit = fitted_parameters[1]
    sigma_fit = fitted_parameters[2]
    slope_fit = fitted_parameters[3]
    intercept_fit = fitted_parameters[4]

    # --------------------------------------------------------
    # Peak-center uncertainty
    # --------------------------------------------------------

    if covariance.shape == (5, 5):

        center_error = np.sqrt(
            max(
                covariance[1, 1],
                0
            )
        )

    else:

        center_error = np.nan

    # --------------------------------------------------------
    # FWHM
    # --------------------------------------------------------

    fwhm_fit = (
        2
        * np.sqrt(2 * np.log(2))
        * sigma_fit
    )

    # --------------------------------------------------------
    # R-squared
    # --------------------------------------------------------

    fitted_y = gaussian_linear(
        x,
        *fitted_parameters
    )

    residual_sum_squares = np.sum(
        (y - fitted_y) ** 2
    )

    total_sum_squares = np.sum(
        (y - np.mean(y)) ** 2
    )

    if total_sum_squares > 0:

        r_squared = (
            1
            - residual_sum_squares
            / total_sum_squares
        )

    else:

        r_squared = np.nan

    # --------------------------------------------------------
    # Store fitted results
    # --------------------------------------------------------

    results.append(
        {
            "run": run,
            "scan_index": scan_index,
            "label": scan_label,
            "file_path": file_path,
            "scan_time": scan_time,
            "peak_position": center_fit,
            "peak_error": center_error,
            "fwhm": fwhm_fit,
            "r_squared": r_squared
        }
    )

    print(
        f"{scan_label}: "
        f"peak = {center_fit:.6f}°, "
        f"FWHM = {fwhm_fit:.6f}°, "
        f"R² = {r_squared:.5f}"
    )

    # --------------------------------------------------------
    # Show individual peak fit
    # --------------------------------------------------------

    if show_individual_fits:

        x_dense = np.linspace(
            fit_min,
            fit_max,
            500
        )

        y_dense = gaussian_linear(
            x_dense,
            *fitted_parameters
        )

        background_dense = (
            slope_fit * x_dense
            + intercept_fit
        )

        fig_fit, ax_fit = plt.subplots(
            figsize=(7, 5)
        )

        ax_fit.scatter(
            x,
            y,
            s=30,
            label="Experimental data",
            zorder=3
        )

        ax_fit.plot(
            x_dense,
            y_dense,
            linewidth=2,
            label="Gaussian fit"
        )

        ax_fit.plot(
            x_dense,
            background_dense,
            linestyle="--",
            linewidth=1.5,
            label="Linear background"
        )

        ax_fit.axvline(
            center_fit,
            linestyle=":",
            linewidth=1.5,
            label=(
                f"Peak center = "
                f"{center_fit:.5f}°"
            )
        )

        ax_fit.set_xlim(
            fit_min,
            fit_max
        )

        ax_fit.set_xlabel(
            r"2$\theta$ (°)",
            fontsize=16
        )

        ax_fit.set_ylabel(
            "Intensity",
            fontsize=16
        )

        ax_fit.tick_params(
            axis="both",
            labelsize=14
        )

        ax_fit.legend(
            frameon=False,
            fontsize=11
        )

        ax_fit.set_title(
            f"{scan_label}, $R^2$ = {r_squared:.4f}",
            fontsize=14
        )

        fig_fit.tight_layout()
        plt.show()


if len(results) == 0:

    raise RuntimeError(
        "No CoO peaks were successfully fitted."
    )


# ============================================================
# SORT SUCCESSFUL FITS BY TIME
# ============================================================

if all(
    item["scan_time"] is not None
    for item in results
):

    results.sort(
        key=lambda item: item["scan_time"]
    )


# ============================================================
# CALCULATE ELAPSED TIME
# ============================================================

if all(
    item["scan_time"] is not None
    for item in results
):

    reference_time = results[0]["scan_time"]

    elapsed_time = np.array(
        [
            (
                item["scan_time"]
                - reference_time
            ).total_seconds()
            for item in results
        ],
        dtype=float
    )

else:

    elapsed_time = (
        np.arange(len(results))
        * seconds_per_scan
    )


peak_positions = np.array(
    [
        item["peak_position"]
        for item in results
    ],
    dtype=float
)

peak_errors = np.array(
    [
        item["peak_error"]
        for item in results
    ],
    dtype=float
)


# ============================================================
# PRINT FINAL ORDER
# ============================================================

print("\nFinal scan order:")

for elapsed, item in zip(
    elapsed_time,
    results
):

    print(
        f"{item['label']}: "
        f"time = {elapsed:.1f} s, "
        f"peak = {item['peak_position']:.6f}°"
    )


# ============================================================
# PLOT PEAK POSITION VS ELAPSED TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=figure_size
)

line, = ax.plot(
    elapsed_time,
    peak_positions,
    linewidth=line_width
)

ax.scatter(
    elapsed_time,
    peak_positions,
    s=marker_size,
    color=line.get_color(),
    zorder=3
)

# Plot fitting uncertainties when available
finite_error = np.isfinite(
    peak_errors
)

if np.any(finite_error):

    ax.errorbar(
        elapsed_time[finite_error],
        peak_positions[finite_error],
        yerr=peak_errors[finite_error],
        fmt="none",
        color=line.get_color(),
        capsize=3,
        linewidth=1
    )

ax.set_xlabel(
    "Elapsed time at 900 °C (s)",
    fontsize=20
)

ax.set_ylabel(
    r"CoO peak position, 2$\theta$ (°)",
    fontsize=20
)

ax.tick_params(
    axis="both",
    labelsize=18
)

ax.grid(False)

fig.tight_layout()


# ============================================================
# SAVE FINAL RESULT
# ============================================================

if save_result:

    output_figure = os.path.join(
        base_dir,
        "CoO_peak_position_900C_C1.png"
    )

    fig.savefig(
        output_figure,
        dpi=output_dpi,
        bbox_inches="tight"
    )

    output_text = os.path.join(
        base_dir,
        "CoO_peak_position_900C_C1.txt"
    )

    scan_indices = np.array(
        [
            item["scan_index"]
            for item in results
        ],
        dtype=int
    )

    output_array = np.column_stack(
        [
            elapsed_time,
            peak_positions,
            peak_errors,
            scan_indices
        ]
    )

    np.savetxt(
        output_text,
        output_array,
        header=(
            "Elapsed_time_s "
            "CoO_peak_position_2theta_deg "
            "Peak_position_standard_error_deg "
            "Scan_index"
        ),
        fmt=[
            "%.2f",
            "%.8f",
            "%.8f",
            "%d"
        ]
    )

    print(
        f"\nSaved figure:\n{output_figure}"
    )

    print(
        f"\nSaved fitted peak positions:\n"
        f"{output_text}"
    )


plt.show()