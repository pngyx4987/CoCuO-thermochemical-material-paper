# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:41:11 2026

@author: pyx
"""

import matplotlib.pyplot as plt

# Temperature profile
time = [
    0, 5, 20, 104.5, 107, 198.8, 201.3, 305, 306.5, 344.3,
    345.8, 362.9, 365.4, 394.2, 396.7, 426.1, 427.6,
    503.1, 507.1, 556.5, 559.4166667, 564.4166667
]

temperature = [
    25, 25, 750, 750, 900, 900, 800, 800, 750, 750,
    800, 800, 900, 900, 800, 800, 750, 750, 900, 900,
    25, 25
]

# Points to overlay
point_time = [
    110, 116.55, 124.6833333, 131.2166667,204.3,217,229.7,253.3,
    311.5, 318.45, 327.9166667, 334.8,
    368.4, 373.9333333, 380.0166667, 391.2333333,
    434.15, 440.5333333, 446.7166667, 472.8833333
]

point_temp = [
    900, 900, 900, 900,
    800,800,800,800,
    750, 750, 750, 750,
    900, 900, 900, 900,
    750, 750, 750, 750
]

# Plot
plt.figure(figsize=(10, 5))

# Temperature profile
plt.step(time, temperature, where='post',
         color='black', linewidth=2)

# Scatter points
plt.scatter(point_time, point_temp,
            color='red', s=60, marker='o',
            zorder=3)

# Labels
plt.xlabel("Time (min)", fontsize=18)
plt.ylabel("Temperature (°C)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.xlim(0, 570)
plt.ylim(0, 950)
plt.legend()

plt.tight_layout()
plt.show()