# -*- coding: utf-8 -*-
"""
Driver script for dist_func_z_evol.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import dist_func, magnetic_equilibrium, build_flux_tube_geometry, dist_func_z_evol

# %% Plot f at z-positions
# Plot the distribution function at 2 z-positions

#########################################################################################
#### HTPD 2026 Plot
vel, xsi, f = dist_func(n_fast = 5e18,
                        n_max = 5e19,
                        T_max = 1,
                        R_m = 57,
                        E_NBI = 25,
                        theta_NBI = np.pi/4,
                        T_e = 0.075,
                        mu_i = 2,
                        Z_eff = 3,
                        gridsize = 500)

# Location of magnetic equilibrium file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

# Load the magnetic equilibrium
Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk)

# Radial distance where the distribution function is calculated from the midplane
rDist = 0.01 # [m]

# z-positions where we want the evolved distribution function
zValArr = np.linspace(0, 0.8, 20)

# Precompute the flux tube geometry (B/B0 along the tube) for this single
# starting radius, then evolve f along z with mu-conservation
rAlongTube_all, bNormArr_all, RmArr = build_flux_tube_geometry(np.array([rDist]), zValArr, Rmesh, Zmesh, Bmag, magneticFlux)
zEvolF = dist_func_z_evol(f, vel, xsi, bNormArr_all[0], zValArr=zValArr, makeplot=True)
