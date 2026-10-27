# -*- coding: utf-8 -*-
"""
Driver script for detector_angle_optimization.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing detector_instrument_function.py.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector_instrument_function import detector_angle_optimization

# %% Find the optimal angle for a given detector position

filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
filenameEqdsk = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

# Actual (XYZ) location of the detector
detPos = np.array([-25.905, -0.235, 42.6]) * 1e-2

detRad = 0.5 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.3

tubeAng = 10 * np.pi/180

optimalAngle = detector_angle_optimization(detPos, detSize, bendRad, tubeAng, filenameEqdsk, makeplot=True)

# %% Find the optimal angle for all of the detectors

filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
filenameEqdsk = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

savename = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing.npz'
savename = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing_whamr.npz'
# savename = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_whamr_cdr.npz'

# Positions of the detectors for the SW 12in CF flange
# zDelta = 1.7 * 25.4 * 1e-3 # 1.7in apart
# zPosArr = -np.arange(0.323, 0.46, zDelta)
# xPos = 398 * 1e-3
# yPos = -12 * 1e-3

# Possible positions for the future array
zDelta = 0.05 # 5cm apart
zPosArr = np.arange(0.3, 0.65, zDelta)
xPos = 398 * 1e-3
yPos = -12 * 1e-3

detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

detRad = 1.0 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.3

tubeAng = 10 * np.pi/180

optimalAngleArr = np.zeros_like(zPosArr)

for i in range(len(detPosArr)):

    print(f'Optimizing detector {i+1} of {len(detPosArr)}')

    optimalAngleArr[i] = detector_angle_optimization(detPosArr[i], detSize, bendRad, tubeAng, filenameEqdsk)

# Save the full detector geometry alongside the optimal angles so that anything
# loading this file uses the same geometry the angles were optimized for
np.savez(file = savename,
         zPosArr = zPosArr,
         optimalAngleArr = optimalAngleArr, # degrees
         detPosArr = detPosArr, # meters
         detSizeArr = np.full(len(zPosArr), detSize), # m^2
         bendRadArr = np.full(len(zPosArr), bendRad), # meters
         tubeAngArr = np.full(len(zPosArr), tubeAng)) # radians

# Plot the optimal angles
fig = plt.figure(figsize=(12, 8), tight_layout=True)
ax = fig.add_subplot(111)

ax.scatter(zPosArr, optimalAngleArr, s=200)
ax.plot(zPosArr, optimalAngleArr, linewidth=3)

ax.set_title('Optimal Angles for detectors')
ax.set_xlabel('Z [m]')
ax.set_ylabel('Angle [deg]')

plt.show()
