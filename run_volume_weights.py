# -*- coding: utf-8 -*-
"""
Driver script for volume_weights.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing detector_instrument_function.py.
"""

import numpy as np

from detector_instrument_function import b_field_interpolation, volume_weights

# %% Plot normalized detector response function
# Used to generate and plot the normalized detector response function for a
# given magnetic equilibrium and detector geometry.
#
# This is useful for visualizing the response function and making sure it
# looks reasonable.

detPos = np.array([-0.257,  0.307,  0.5])
detPhi = (np.pi/180) * (280)
detRad = 0.5 # inches
detSize = (np.pi*detRad*detRad) / 1550 # m^2
bendRad = 0.7 # meters
tubeAng = 10 * np.pi/180 # radians

filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
b_field_interpolation(filenameEqdsk)

_, _, _, _, _ = volume_weights(detPos, detPhi, detSize, bendRad, tubeAng,
                               cellSize=1e-2, errorLim=None, maxParticles=500,
                               makeplot=True, savename='volume_weights_0.7m_10deg')
