# -*- coding: utf-8 -*-
"""
Driver script for dist_func.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import dist_func

# %% Plot f at midplane
# Plot the distribution function at the midplane

#########################################################################################
#### HTPD 2026 Plot
# Save a plot of the distribution function at z=0 for a 0.1keV Te case
_, _, _ = dist_func(n_fast = 5e18,
                    n_max = 5e19,
                    T_max = 0.03,
                    R_m = 57,
                    E_NBI = 25,
                    theta_NBI = np.pi/4,
                    T_e = 0.075,
                    mu_i = 2,
                    Z_eff = 3,
                    gridsize = 500,
                    makeplot = True,
                    savename = 'dist_func_newbasis.png')
#########################################################################################
