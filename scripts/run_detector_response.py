# -*- coding: utf-8 -*-
"""
Driver script for generate_detector_response.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing detector_instrument_function.py.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector_instrument_function import generate_detector_response

# %% Save normalized detector response function
# Used to generate the detector response for a given magnetic equilibrium,
# fusion reactivity profile and detector geometry.
#
# The output is saved as a .npz file in the reactivity folder.

##############################################################################
#### Detector positions (boxport)

# The detectors are evenly spaced on the boxport
zPosArr = np.arange(0.362, 0.665, 2*2.54/1e2) # 2 in apart on the boxport
xPos = -0.257
yPos = 0.307
detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

# Detector angles
detPhiArr = np.array([282, 278, 275, 271, 264, 258]) * (np.pi/180)
##############################################################################

##############################################################################
#### Detector positions (HTPD 2026)
zPosArr = np.arange(0.2, 0.61, 0.05)
xPos = -0.257
yPos = 0.307
detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

# Detector angles
detPhiArr = np.array([296, 294, 292, 290, 288, 286, 284, 278, 273]) * (np.pi/180)
##############################################################################

# Detector sizes
detRad = 0.5 # inches
detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

# Tube bend radii
bendRadArr = np.full(len(zPosArr), 1.2) # meters

# Tube sector angle
tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians

#### eqdsk file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

#### Fusion reactivity profile
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/Te_2keV_NBI_2d_reactivity.npz'
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d.npz'
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_with_maxwellian.npz'

# Generate the detector response for the given-
# 1. Magnetic equilibrium
# 2. Reactivity profile
# 3. Detector geometry

detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                         makeplot=True, savename='detector_response_with_maxwellian.npz')

# %% Plot the detector array response for a given reactivity profile
# Calculate and plot the detector response for a given reactivity profile
# and detector geometry.

##############################################################################
# Detector geometry (after optimization)

file = np.load('/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing.npz')

# (x,y,z) positions. These are mounted on the CC inner wall.
zPosArr = file['zPosArr'] # meters
xPos = -0.257
yPos = 0.307
detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

# Detector angles (after optimization)
detPhiArr = file['optimalAngleArr'] * (np.pi/180) # radians

# Detector sizes
detRad = 1.0 # inches
detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

# Tube bend radii
bendRadArr = np.full(len(zPosArr), 1.2) # meters

# Tube sector angles
tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians
##############################################################################

# 2D reactivity profile
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_10percent_fast_ions.npz'

# eqdsk file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

# Generate the detector response
detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                         makeplot=True, savename='detector_response_with_10percent_fast_ions.npz')

# %% Check effect of different distribution functions
# Check effect of maxwellian and fast ion components on the detector response.

##############################################################################
#### Detector positions (HTPD 2026)
zPosArr = np.arange(0.2, 0.61, 0.05)
xPos = -0.257
yPos = 0.307
detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

# Detector angles
detPhiArr = np.array([292, 292, 288, 288, 286, 284, 280, 276, 268]) * (np.pi/180)
##############################################################################

# Detector sizes
detRad = 0.5 # inches
detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

# Tube bend radii
bendRadArr = np.full(len(zPosArr), 1.2) # meters

# Tube sector angle
tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians

#### eqdsk file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

# Fast and maxwellian
print('Calculating response with both fast and maxwellian components')
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_fast_and_maxwellian.npz'
detResponseFastAndMaxwellian = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                          makeplot=False)

# Only maxwellian
print('Calculating response with only maxwellian component')
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_maxwellian_only.npz'
detResponseMaxwellianOnly = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                       makeplot=False)

# Only fast
print('Calculating response with only fast ion component')
filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_fast_only.npz'
detResponseFastOnly = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                 makeplot=False)

# Plot the 3 responses together
fig = plt.figure(figsize=(12, 8), tight_layout=True)
ax = fig.add_subplot(111)

# Fast and Maxwellian
ax.plot(detPosArr[:, 2], detResponseFastAndMaxwellian/1e3, linewidth=3, label='Fast + Maxwellian')
ax.scatter(detPosArr[:, 2], detResponseFastAndMaxwellian/1e3, s=200)

# Maxwellian only
ax.plot(detPosArr[:, 2], detResponseMaxwellianOnly/1e3, linewidth=3, label='Maxwellian only')
ax.scatter(detPosArr[:, 2], detResponseMaxwellianOnly/1e3, s=200)

# Fast only
ax.plot(detPosArr[:, 2], detResponseFastOnly/1e3, linewidth=3, label='Fast only')
ax.scatter(detPosArr[:, 2], detResponseFastOnly/1e3, s=200)

ax.legend()
ax.set_ylim(0, None)
ax.set_xlabel('Z [m]')
ax.set_ylabel('Response [counts/ms]')

plt.show()
