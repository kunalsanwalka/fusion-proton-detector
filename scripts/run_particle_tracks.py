# -*- coding: utf-8 -*-
"""
Driver script for generate_tracks_aperture.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing detector_instrument_function.py.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector_instrument_function import b_field_interpolation, generate_tracks_aperture

# %% Check particle tracks
# Used to check the angle of a specific detector to make sure it sees the
# core of the plasma.

# WHAM-HTS
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
# WHAM-R
filenameEqdsk = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/1.000/components/eq__pleiades_3/eqdsk'

b_field_interpolation(filenameEqdsk)

detPos = np.array([-0.257,  0.307,  0.0])
detPos = np.array([398, -12, -453]) * 1e-3

detPhi = (np.pi/180) * (160)

detRad = 0.5 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.3

tubeAng = 10 * np.pi/180

# Save the particle tracks
savename = '/home/sanwalka/synthetic_proton_detector/particle_tracks/track_data_0.7m_10deg.pkl'
savename = None

print(detPos, detPhi, detSize, bendRad, tubeAng)

openingTracks = generate_tracks_aperture(detPos, detPhi, detSize, bendRad, tubeAng, makeplot=True, saveplot=False, savename=savename)
