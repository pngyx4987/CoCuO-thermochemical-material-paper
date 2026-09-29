# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 08:11:29 2026

@author: pyx
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile

# ============================================================
# Load stack
# ============================================================
file_path = (
    r"C:\Users\pyx\Desktop\documents\SBU WORK\paper\CuCoO"
    r"\data_process\3DXANES"
    r"\Co_thickness_masked_3D_stack_imp_remove.tif"
)

stack = tifffile.imread(file_path)

print("Stack shape (Z, Y, X):", stack.shape)

# ============================================================
# ROI (ImageJ coordinates)
#
# X: 1-184
# Y: 1-244
# Z: 1-100
#
# Convert to Python indexing
# ============================================================
roi = stack[
    0:100,      # Z = 1-100
    0:244,      # Y = 1-244
    0:184       # X = 1-184
]

# ============================================================
# Rest of the volume
# ============================================================
mask = np.ones(stack.shape, dtype=bool)

mask[
    0:100,
    0:244,
    0:184
] = False

rest = stack[mask]

# ============================================================
# Remove NaN values
# ============================================================
roi = roi[np.isfinite(roi)]
rest = rest[np.isfinite(rest)]

print(f"ROI voxels : {len(roi):,}")
print(f"Rest voxels: {len(rest):,}")

# ============================================================
# Common histogram bins
# ============================================================
xmin = min(roi.min(), rest.min())
xmax = max(roi.max(), rest.max())

bins = np.linspace(xmin, xmax, 80)

# ============================================================
# Plot histogram
# ============================================================
plt.figure(figsize=(8,6))

plt.hist(
    roi,
    bins=bins,
    density=True,
    alpha=0.6,
    label="ROI",
    edgecolor='black'
)

plt.hist(
    rest,
    bins=bins,
    density=True,
    alpha=0.6,
    label="Rest",
    edgecolor='black'
)

plt.xlabel("Co Thickness")
plt.ylabel("Probability Density")

plt.legend()

plt.tight_layout()
plt.show()

# ============================================================
# Statistics
# ============================================================
print("\nROI")
print(f"Mean   : {roi.mean():.4f}")
print(f"Median : {np.median(roi):.4f}")
print(f"Std    : {roi.std():.4f}")

print("\nRest")
print(f"Mean   : {rest.mean():.4f}")
print(f"Median : {np.median(rest):.4f}")
print(f"Std    : {rest.std():.4f}")