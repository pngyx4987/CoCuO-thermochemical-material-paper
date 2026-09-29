# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:49:03 2026

@author: pyx
"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ============================================================
# Input pre-treated TIFF stacks
# ============================================================
co_thickness_file = (
    r"C:\Users\pyx\Desktop\documents\SBU WORK\paper\CuCoO"
    r"\data_process\3DXANES\Co_thickness_masked_3D_stack_imp_remove.tif"
)

co_ratio_file = (
    r"C:\Users\pyx\Desktop\documents\SBU WORK\paper\CuCoO"
    r"\data_process\3DXANES\Co_ratio_thickness_masked_3D_stack_imp_remove.tif"
)

peak_position_file = (
    r"C:\Users\pyx\Desktop\documents\SBU WORK\paper\CuCoO"
    r"\data_process\3DXANES\peak_pos_masked_3D_stack_imp_remove.tif"
)

output_folder = (
    r"C:\Users\pyx\Desktop\documents\SBU WORK\paper\CuCoO"
    r"\data_process\3DXANES\voxel_correlation"
)

os.makedirs(output_folder, exist_ok=True)

# ============================================================
# Plot settings
# ============================================================
max_plot_points = 300_000
random_seed = 0
point_size = 1
point_alpha = 0.15


# ============================================================
# Read TIFF helper
# ============================================================
def read_3d_tiff(filename, name):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{name} file not found:\n{filename}"
        )

    volume = tifffile.imread(filename)
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D volume, "
            f"but found shape {volume.shape}"
        )

    print(
        f"{name}: shape={volume.shape}, "
        f"dtype={volume.dtype}"
    )

    return volume.astype(np.float32)


# ============================================================
# Read the three volumes
# ============================================================
co_thickness = read_3d_tiff(
    co_thickness_file,
    "Co thickness"
)

co_ratio = read_3d_tiff(
    co_ratio_file,
    "Co ratio"
)

peak_position = read_3d_tiff(
    peak_position_file,
    "Peak position"
)


# ============================================================
# Check matching shapes
# ============================================================
if not (
    co_thickness.shape
    == co_ratio.shape
    == peak_position.shape
):
    raise ValueError(
        "The three stacks have different shapes:\n"
        f"Co thickness:  {co_thickness.shape}\n"
        f"Co ratio:      {co_ratio.shape}\n"
        f"Peak position: {peak_position.shape}"
    )


# ============================================================
# Clip peak position values
#
# Values > 7.730 become 7.730
# Values < 7.726 become 7.726
# ============================================================
peak_position = np.clip(
    peak_position,
    7.726,
    7.730
)


# ============================================================
# Keep only voxels that are finite in all three stacks
# ============================================================
valid = (
    np.isfinite(co_thickness)
    & np.isfinite(co_ratio)
    & np.isfinite(peak_position)
)

n_total = valid.size
n_valid = int(np.count_nonzero(valid))

print("\nTotal voxels:", n_total)
print("Valid voxels:", n_valid)
print("Skipped voxels:", n_total - n_valid)


# ============================================================
# Extract corresponding voxel values
# ============================================================
amount_all = co_thickness[valid]
ratio_all = co_ratio[valid]
peak_all = peak_position[valid]


# ============================================================
# Randomly select the same voxels for both plots
# ============================================================
rng = np.random.default_rng(random_seed)

if n_valid > max_plot_points:
    selected = rng.choice(
        n_valid,
        size=max_plot_points,
        replace=False
    )

    amount = amount_all[selected]
    ratio = ratio_all[selected]
    peak = peak_all[selected]

    print(
        f"Plotting {max_plot_points:,} randomly selected "
        f"voxels from {n_valid:,} valid voxels."
    )
else:
    amount = amount_all
    ratio = ratio_all
    peak = peak_all

    print(f"Plotting all {n_valid:,} valid voxels.")


# ============================================================
# Plot 1: Co ratio vs. peak position
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    ratio,
    peak,
    s=point_size,
    alpha=point_alpha,
    rasterized=True
)

ax.set_xlabel("Co ratio")
ax.set_ylabel("Fitted peak position")
ax.set_title("Co ratio vs. fitted peak position")

ax.grid(True, alpha=0.25)

plt.tight_layout()

ratio_figure = os.path.join(
    output_folder,
    "Co_ratio_vs_peak_position.png"
)

plt.savefig(
    ratio_figure,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print(ratio_figure)

plt.show()


# ============================================================
# Plot 2: Co amount/thickness vs. peak position
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    amount,
    peak,
    s=point_size,
    alpha=point_alpha,
    rasterized=True
)

ax.set_xlabel("Co amount / thickness")
ax.set_ylabel("Fitted peak position")
ax.set_title("Co amount vs. fitted peak position")

ax.grid(True, alpha=0.25)

plt.tight_layout()

amount_figure = os.path.join(
    output_folder,
    "Co_amount_vs_peak_position.png"
)

plt.savefig(
    amount_figure,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print(amount_figure)

plt.show()
# ============================================================
# Plot 3: 3D scatter
#
# x = Co amount / thickness
# y = Co ratio
# z = fitted peak position
#
# The arrays already exclude:
# - NaN / infinity in any stack
# - peak position < 7.726
# - peak position > 7.730
# ============================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    amount,
    ratio,
    peak,
    c=peak,
    s=point_size,
    alpha=point_alpha,
    rasterized=True
)

ax.set_xlabel("Co amount / thickness")
ax.set_ylabel("Co ratio")
ax.set_zlabel("Fitted peak position")

ax.set_zlim(7.726, 7.730)

ax.set_title(
    "Co amount, Co ratio, and fitted peak position"
)

cbar = fig.colorbar(
    scatter,
    ax=ax,
    pad=0.12,
    shrink=0.7
)

cbar.set_label("Fitted peak position")

plt.tight_layout()

scatter_3d_file = os.path.join(
    output_folder,
    "Co_amount_ratio_peak_position_3D.png"
)

plt.savefig(
    scatter_3d_file,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print(scatter_3d_file)

plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Keep voxels finite in thickness and peak-position data
# ============================================================
valid = (
    np.isfinite(co_thickness)
    & np.isfinite(peak_position)
)

thickness = co_thickness[valid]
peak = peak_position[valid]

print("Valid voxels:", len(thickness))


# ============================================================
# Define peak-position groups
#
# The group label is the upper boundary:
#
# 7.726: peak <= 7.726
# 7.727: 7.726 < peak <= 7.727
# 7.728: 7.727 < peak <= 7.728
# 7.729: 7.728 < peak <= 7.729
# 7.730: peak > 7.729
#
# Therefore, values below 7.726 are included in the first
# group, and values above 7.730 are included in the last group.
# ============================================================
peak_groups = [
    {
        "label": "≤ 7.726",
        "mask": peak <= 7.727
    },

    {
        "label": "7.727–7.728",
        "mask": (peak > 7.727) & (peak <= 7.728)
    },
    {
        "label": "7.728–7.729",
        "mask": (peak > 7.728) & (peak <= 7.729)
    },
    {
        "label": "≥ 7.729",
        "mask": peak > 7.729
    }
]


# ============================================================
# Use shared histogram bins so that all distributions can be
# compared directly.
#
# The 0.5 and 99.5 percentiles prevent a few extreme thickness
# values from stretching the histogram range.
# ============================================================
thickness_min = np.nanpercentile(thickness, 0.5)
thickness_max = np.nanpercentile(thickness, 99.5)

bins = np.linspace(
    thickness_min,
    thickness_max,
    81
)


# ============================================================
# Plot Co-thickness distributions
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for group in peak_groups:

    group_thickness = thickness[group["mask"]]

    print(
        f"{group['label']}: "
        f"{len(group_thickness):,} voxels"
    )

    if len(group_thickness) == 0:
        continue

    ax.hist(
        group_thickness,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=group["label"]
    )

ax.set_xlabel("Co thickness")
ax.set_ylabel("Probability density")

ax.set_title(
    "Co thickness distribution by fitted peak-position range"
)

ax.legend(
    title="Peak position",
    frameon=False
)


plt.tight_layout()


# ============================================================
# Save figure
# ============================================================
output_file = os.path.join(
    output_folder,
    "Co_thickness_distribution_by_peak_position_range.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(output_file)

plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Keep voxels finite in ratio and peak-position data
# ============================================================
valid = (
    np.isfinite(co_ratio)
    & np.isfinite(peak_position)
)

ratio = co_ratio[valid]
peak = peak_position[valid]

print("Valid voxels:", len(ratio))


# ============================================================
# Define peak-position groups
#
# The group label is the upper boundary:
#
# 7.726: peak <= 7.726
# 7.727: 7.726 < peak <= 7.727
# 7.728: 7.727 < peak <= 7.728
# 7.729: 7.728 < peak <= 7.729
# 7.730: peak > 7.729
#
# The final group includes values above 7.730.
# ============================================================
peak_groups = [
    {
        "label": "≤ 7.726",
        "mask": peak <= 7.726
    },
    {
        "label": "7.726–7.727",
        "mask": (peak > 7.726) & (peak <= 7.727)
    },
    {
        "label": "7.727–7.728",
        "mask": (peak > 7.727) & (peak <= 7.728)
    },
    {
        "label": "7.728–7.729",
        "mask": (peak > 7.728) & (peak <= 7.729)
    },
    {
        "label": "≥ 7.729",
        "mask": peak > 7.729
    }
]


# ============================================================
# Shared histogram bins
# ============================================================
ratio_min = np.nanpercentile(ratio, 0.5)
ratio_max = np.nanpercentile(ratio, 99.5)

bins = np.linspace(
    ratio_min,
    ratio_max,
    81
)


# ============================================================
# Plot ratio distributions
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for group in peak_groups:

    group_ratio = ratio[group["mask"]]

    print(
        f"{group['label']}: "
        f"{len(group_ratio):,} voxels"
    )

    if len(group_ratio) == 0:
        continue

    ax.hist(
        group_ratio,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        
    )

ax.set_xlabel("Co ratio")
ax.set_ylabel("Probability density")
#ax.set_title(
    #"Co ratio distribution by fitted peak-position range"
#)




plt.tight_layout()


# ============================================================
# Save figure
# ============================================================
output_file = os.path.join(
    output_folder,
    "Co_ratio_distribution_by_peak_position_range.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved:")
print(output_file)

plt.show()