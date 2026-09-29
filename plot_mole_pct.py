# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:58:12 2026

@author: pyx
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER INPUT
# ============================================================

base_dir = (
    r"C:/Users/pyx/Desktop/documents/SBU WORK/paper/"
    r"CuCoO/data_process/XPD"
)

file_names = [
    "C1_775C_mole_percent.txt",
    "C1_900C_mole_percent.txt",
    "C2_775C_mole_percent.txt",
    "C2_900C_mole_percent.txt",
    "C3_775C_mole_percent.txt",
    "C3_900C_mole_percent.txt",
]

# Save figures automatically
save_figures = False

# Output image format
figure_format = "png"

# Figure quality
figure_dpi = 300

# Plot appearance
figure_size = (8, 4)
line_width = 1.8
marker_size = 45


# ============================================================
# READ AND PLOT EACH FILE SEPARATELY
# ============================================================

for file_name in file_names:

    file_path = os.path.join(
        base_dir,
        file_name
    )

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # --------------------------------------------------------
    # Read the file
    #
    # names=True:
    # Use the first row as column names
    #
    # delimiter=None:
    # Automatically handle spaces or tabs
    # --------------------------------------------------------

    data = np.genfromtxt(
        file_path,
        names=True,
        delimiter=None,
        encoding="utf-8"
    )

    # Handle the case where the file contains only one data row
    data = np.atleast_1d(data)

    column_names = data.dtype.names

    if column_names is None or len(column_names) < 2:
        print(
            f"Skipping {file_name}: "
            "at least two columns are required."
        )
        continue

    print(f"\nReading: {file_name}")
    print("Columns:", column_names)

    # First column is treated as time
    time_column = column_names[0]
    time = np.asarray(
        data[time_column],
        dtype=float
    )

    # Remaining columns are phase mole percentages
    phase_columns = column_names[1:]

    # --------------------------------------------------------
    # Sort data by time
    # --------------------------------------------------------

    valid_time = np.isfinite(time)

    time = time[valid_time]

    sort_order = np.argsort(time)
    time = time[sort_order]

    # --------------------------------------------------------
    # Create a separate figure for this file
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    for phase_name in phase_columns:

        mole_percent = np.asarray(
            data[phase_name],
            dtype=float
        )

        mole_percent = mole_percent[valid_time]
        mole_percent = mole_percent[sort_order]

        valid_phase = np.isfinite(mole_percent)

        phase_time = time[valid_phase]
        phase_value = mole_percent[valid_phase]

        if len(phase_time) == 0:
            print(
                f"No valid data for phase "
                f"{phase_name} in {file_name}"
            )
            continue

        # Plot connecting line
        line, = ax.plot(
            phase_time,
            phase_value,
            linewidth=line_width,
            label=phase_name
        )

        # Plot scatter points using the same color as the line
        ax.scatter(
            phase_time,
            phase_value,
            s=marker_size,
            color=line.get_color(),
            zorder=3
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel(
        "",
        fontsize=22
    )

    ax.set_ylabel(
        "",
        fontsize=22
    )

    ax.tick_params(
        axis="both",
        labelsize=22
    )

    ax.set_ylim(
        0,100
    )
    
    ax.set_xlim(
        0,1300
    )

    ax.grid(False)

    ax.legend(
        fontsize=10,
        frameon=False
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    if save_figures:

        output_name = (
            os.path.splitext(file_name)[0]
            + f".{figure_format}"
        )

        output_path = os.path.join(
            base_dir,
            output_name
        )

        fig.savefig(
            output_path,
            dpi=figure_dpi,
            bbox_inches="tight"
        )

        print(f"Saved: {output_path}")

    plt.show()