# -*- coding: utf-8 -*-
"""
Driver script for generate_tracks_aperture.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing detector_instrument_function.py.
"""

import numpy as np

from detector_instrument_function import b_field_interpolation, generate_tracks_aperture

# %% Check particle tracks
# Used to check the angle of a specific detector to make sure it sees the
# core of the plasma.

filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

b_field_interpolation(filenameEqdsk)

detPos = np.array([-0.257,  0.307,  0.0])

detPhi = (np.pi/180) * (296)

detRad = 0.5 # inches
detSize = (np.pi*detRad*detRad) / 1550

bendRad = 0.7

tubeAng = 10 * np.pi/180

# Save the particle tracks
savename = '/home/sanwalka/synthetic_proton_detector/particle_tracks/track_data_0.7m_10deg.pkl'

openingTracks = generate_tracks_aperture(detPos, detPhi, detSize, bendRad, tubeAng, makeplot=True, saveplot=False, savename=savename)
