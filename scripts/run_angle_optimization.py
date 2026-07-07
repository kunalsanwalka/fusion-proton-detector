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

detPos = np.array([-0.257,  0.307,  0.5])

detRad = 0.5 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.7

tubeAng = 10 * np.pi/180

optimalAngle = detector_angle_optimization(detPos, detSize, bendRad, tubeAng, filenameEqdsk, makeplot=True)

# %% Find the optimal angle for all of the detectors

filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

zPosArr = np.arange(0.0, 0.61, 0.05)
xPos = -0.257
yPos = 0.307
detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

detRad = 1.0 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.7

tubeAng = 10 * np.pi/180

optimalAngleArr = np.zeros_like(zPosArr)

for i in range(len(detPosArr)):

    print(f'Optimizing detector {i+1} of {len(detPosArr)}')

    optimalAngleArr[i] = detector_angle_optimization(detPosArr[i], detSize, bendRad, tubeAng, filenameEqdsk)

np.savez(file = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing.npz',
         zPosArr = zPosArr,
         optimalAngleArr = optimalAngleArr)

# Plot the optimal angles
fig = plt.figure(figsize=(12, 8), tight_layout=True)
ax = fig.add_subplot(111)

ax.scatter(zPosArr, optimalAngleArr, s=200)
ax.plot(zPosArr, optimalAngleArr, linewidth=3)

ax.set_title('Optimal Angles for detectors')
ax.set_xlabel('Z [m]')
ax.set_ylabel('Angle [deg]')

plt.show()
