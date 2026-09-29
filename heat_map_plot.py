# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:54:27 2026

@author: pyx
"""
from pathlib import Path
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. User settings
# ============================================================
directory = Path(
    r"C:/Users/pyx/Desktop/documents/SBU WORK/DEGREE/"
    r"XPD_CuO_Co3O4/refinement_CuCoO/"
    r"baseline_corrected - YXP"
)

seconds_per_scan = 30

# Displayed 2θ range
x_min = 1
x_max = 8

# Heatmap color range
intensity_min = 5
intensity_max = 200


# ============================================================
# 2. Extract scan time from filename
# ============================================================
def extract_scan_time(file_path):
    """
    Extract the timestamp and scan index from a filename containing:

        YYYYMMDD-HHMMSS-index

    Example:
        2024_XPD_C3_Co3O4-CuO_2_
        20241006-183416-2_baseline_corrected_RT.xy

    Base time:
        2024-10-06 18:34:16

    Scan index:
        2

    Actual scan time:
        2024-10-06 18:35:16
    """
    filename = file_path.stem

    match = re.search(
        r"(?<!\d)(\d{8})-(\d{6})-(\d+)(?!\d)",
        filename,
    )

    if match is None:
        raise ValueError(
            f"Could not extract scan information from:\n"
            f"{file_path.name}"
        )

    date_string = match.group(1)
    time_string = match.group(2)
    scan_index = int(match.group(3))

    base_time = datetime.strptime(
        date_string + time_string,
        "%Y%m%d%H%M%S",
    )

    actual_scan_time = base_time + timedelta(
        seconds=scan_index * seconds_per_scan
    )

    return actual_scan_time


# ============================================================
# 3. Find .xy files and remove files containing NA
# ============================================================
all_files = list(
    directory.rglob("*.xy")
)

if len(all_files) == 0:
    raise FileNotFoundError(
        f"No .xy files were found in:\n{directory}"
    )

xy_files = [
    file_path
    for file_path in all_files
    if "NA" not in file_path.name.upper()
]

print("Total .xy files found:", len(all_files))
print("Files containing NA skipped:", len(all_files) - len(xy_files))
print("Files remaining:", len(xy_files))

if len(xy_files) == 0:
    raise FileNotFoundError(
        "No valid .xy files remained after excluding NA files."
    )


# ============================================================
# 4. Read scan times and skip unrecognized filenames
# ============================================================
file_time_pairs = []

for file_path in xy_files:
    try:
        actual_scan_time = extract_scan_time(file_path)

        file_time_pairs.append(
            (
                file_path,
                actual_scan_time,
            )
        )

    except ValueError as error:
        print(error)
        print("This file will be skipped.\n")


if len(file_time_pairs) == 0:
    raise RuntimeError(
        "No filenames matched the expected "
        "YYYYMMDD-HHMMSS-index format."
    )


# Sort according to the calculated actual scan time
file_time_pairs.sort(
    key=lambda item: item[1]
)

xy_files_sorted = [
    item[0]
    for item in file_time_pairs
]

scan_times = [
    item[1]
    for item in file_time_pairs
]


# ============================================================
# 5. Calculate elapsed time
# ============================================================
reference_time = scan_times[0]

elapsed_times = np.array(
    [
        (scan_time - reference_time).total_seconds()
        for scan_time in scan_times
    ],
    dtype=float,
)

print("\nFirst scan:")
print("File:", xy_files_sorted[0].name)
print("Calculated time:", scan_times[0])

print("\nLast scan:")
print("File:", xy_files_sorted[-1].name)
print("Calculated time:", scan_times[-1])

print(
    "\nElapsed time:",
    elapsed_times[0],
    "to",
    elapsed_times[-1],
    "seconds",
)


# ============================================================
# 6. Print the first few calculated times for checking
# ============================================================
print("\nFirst 10 calculated scan times:")

for i in range(
    min(10, len(xy_files_sorted))
):
    print(
        f"{i:4d} | "
        f"{scan_times[i]} | "
        f"elapsed = {elapsed_times[i]:8.1f} s | "
        f"{xy_files_sorted[i].name}"
    )


# ============================================================
# 7. Create the common 2θ grid
# ============================================================
first_data = np.loadtxt(
    xy_files_sorted[0]
)

if first_data.ndim != 2 or first_data.shape[1] < 2:
    raise ValueError(
        "The first file does not contain at least two columns:\n"
        f"{xy_files_sorted[0]}"
    )

x_grid = first_data[:, 0].astype(float)

# Remove nonfinite values
x_grid = x_grid[
    np.isfinite(x_grid)
]

# Ensure increasing 2θ
x_grid = np.sort(x_grid)


# ============================================================
# 8. Read and interpolate the intensity data
# ============================================================
y_matrix = []
valid_elapsed_times = []
valid_files = []

for file_index, file_path in enumerate(xy_files_sorted):

    try:
        data = np.loadtxt(file_path)

        if data.ndim != 2 or data.shape[1] < 2:
            print(
                "Skipping incorrectly formatted file:",
                file_path.name,
            )
            continue

        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)

        # Keep finite x and y values
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            print(
                "Skipping file with insufficient valid data:",
                file_path.name,
            )
            continue

        # Sort by increasing 2θ
        order = np.argsort(x)

        x = x[order]
        y = y[order]

        # Remove duplicate 2θ positions
        x, unique_indices = np.unique(
            x,
            return_index=True,
        )

        y = y[unique_indices]

        # Interpolate raw stored intensity onto the common grid
        y_interpolated = np.interp(
            x_grid,
            x,
            y,
            left=np.nan,
            right=np.nan,
        )

        y_matrix.append(
            y_interpolated
        )

        valid_elapsed_times.append(
            elapsed_times[file_index]
        )

        valid_files.append(
            file_path
        )

    except Exception as error:
        print(
            f"Skipping {file_path.name} because of error:"
        )
        print(error)


if len(y_matrix) == 0:
    raise RuntimeError(
        "No valid scans remained after loading the files."
    )

y_matrix = np.asarray(
    y_matrix,
    dtype=float,
)

valid_elapsed_times = np.asarray(
    valid_elapsed_times,
    dtype=float,
)

print("\nNumber of scans plotted:", len(valid_files))
print("Heatmap matrix shape:", y_matrix.shape)
print("Minimum intensity:", np.nanmin(y_matrix))
print("Maximum intensity:", np.nanmax(y_matrix))


# ============================================================
# 9. Select the desired 2θ range
# ============================================================
x_mask = (
    (x_grid >= x_min)
    & (x_grid <= x_max)
)

x_plot = x_grid[x_mask]

y_plot = y_matrix[
    :,
    x_mask
]

if len(x_plot) < 2:
    raise ValueError(
        f"The selected range {x_min}–{x_max} degrees "
        "contains fewer than two data points."
    )


# ============================================================
# 10. Convert center coordinates into bin edges
# ============================================================
def centers_to_edges(values, default_width=1.0):
    """
    Convert center coordinates to bin-edge coordinates.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 1:
        return np.array(
            [
                values[0] - default_width / 2,
                values[0] + default_width / 2,
            ]
        )

    differences = np.diff(values)

    if np.any(differences <= 0):
        raise ValueError(
            "Coordinate values must be strictly increasing."
        )

    edges = np.empty(
        len(values) + 1,
        dtype=float,
    )

    edges[1:-1] = (
        values[:-1]
        + values[1:]
    ) / 2

    edges[0] = (
        values[0]
        - differences[0] / 2
    )

    edges[-1] = (
        values[-1]
        + differences[-1] / 2
    )

    return edges


x_edges = centers_to_edges(
    x_plot
)

time_edges = centers_to_edges(
    valid_elapsed_times,
    default_width=seconds_per_scan,
)


# ============================================================
# Plot each scan at its exact elapsed time
# ============================================================
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


scan_duration = 30.0  # seconds

fig, ax = plt.subplots(figsize=(10, 7))

# Make regions with no scans black
ax.set_facecolor("black")

# Common color scale for all scans
norm = Normalize(
    vmin=intensity_min,
    vmax=intensity_max,
)

cmap = plt.get_cmap("hot")


# Plot every scan as a 30-second strip
for i in range(len(valid_elapsed_times)):

    scan_start = valid_elapsed_times[i]
    scan_end = scan_start + scan_duration

    ax.imshow(
        y_plot[i:i + 1, :],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[
            x_plot[0],
            x_plot[-1],
            scan_start,
            scan_end,
        ],
    )


# Create the colorbar
colorbar_source = ScalarMappable(
    norm=norm,
    cmap=cmap,
)

colorbar_source.set_array([])

colorbar = fig.colorbar(
    colorbar_source,
    ax=ax,
    pad=0.02,
)

colorbar.set_label(
    "Intensity (arb. u.)",
    fontsize=20,
)

colorbar.ax.tick_params(
    labelsize=18
)


# Axis settings
ax.set_xlabel(
    r"2$\theta$ (degrees)",
    fontsize=20,
)

ax.set_ylabel(
    "Elapsed time (s)",
    fontsize=20,
)

ax.tick_params(
    axis="both",
    labelsize=18,
)

ax.set_xlim(
    x_min,
    x_max,
)

ax.set_ylim(
    0,
    valid_elapsed_times[-1] + scan_duration,
)

fig.tight_layout()
plt.show()