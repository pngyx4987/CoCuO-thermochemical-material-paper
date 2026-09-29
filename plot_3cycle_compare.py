# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:28:48 2026

@author: pyx
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==== USER INPUT ====
run_numbers = ["184652-0","192720-9","201201-9","205246-9"]

# ==== BASE DIRECTORY ====
base_dir = r"C:/Users/pyx/Desktop/documents/SBU WORK/DEGREE/XPD_CuO_Co3O4/refinement_CuCoO/baseline_corrected - YXP"

# ==== PLOT ====
plt.figure(figsize=(8, 7))

for run in run_numbers:
    pattern = os.path.join(base_dir, f"*{run}*.xy")
    matched_files = glob.glob(pattern)
    
    if not matched_files:
        print(f"No file found for run {run}")
        continue
    
    # 👉 Loop over ALL matched files
    for i, file_path in enumerate(matched_files):
        data = np.loadtxt(file_path)
        x = data[:, 0]
        y = data[:, 1]
        
        # Optional: distinguish multiple files with same run
        label = f"{run}" if len(matched_files) == 1 else f"{run}_{i+1}"
        
        if run == "184652-0":
            plt.plot(
                x,
                y,
                linestyle="--",
                color="black",
                linewidth=2,
                label=label,
            )
        else:
            plt.plot(
                x,
                y,
                label=label,
            )        

# ==== FORMATTING ====
plt.xlabel("2θ (degree)", fontsize=20)
plt.ylabel("Intensity (arb.u.)",fontsize=20)
plt.legend(title="Run")
plt.xlim(3.5,5)
plt.ylim(0,40)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(False)

plt.tight_layout()
plt.show()