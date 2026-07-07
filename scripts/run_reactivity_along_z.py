# -*- coding: utf-8 -*-
"""
Driver script for fusion_reactivity along a flux tube.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import (dist_func, magnetic_equilibrium, build_flux_tube_geometry,
                                          dist_func_z_evol, fusion_cross_section, fusion_reactivity,
                                          build_fusion_kernel)

# %% Calculate the reactivity along z for a given flux tube

# =============================================
# Test with a z-evolved maxwellian
# =============================================

# Generate a maxwellian at the midplane with 10keV
vel, xsi, f = dist_func(n_fast = 5,
                        n_max = 5e19,
                        T_max = 10,
                        R_m = 57,
                        E_NBI = 25,
                        theta_NBI = np.pi/4,
                        T_e = 0.075,
                        mu_i = 2,
                        Z_eff = 3,
                        gridsize = 50)

vel_1d = vel[0, :]
xsi_1d = xsi[:, 0]

# Location of magnetic equilibrium file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
# Load the magnetic equilibrium
Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk)

# z-positions where we want to calculate the reactivity
zArr = np.linspace(0, 0.8, 50)
# r-position at the midplane
rMid = 0.06 # m

# Evolve f along z
rAlongTube_all, bNormArr_all, RmArr = build_flux_tube_geometry(np.array([rMid]), zArr, Rmesh, Zmesh, Bmag, magneticFlux)
zEvolF = dist_func_z_evol(f, vel, xsi, bNormArr_all[0])

reactivity1D = np.zeros(shape=len(zArr))

# Calculate the reactivity at each z position
fusion_cross_section()
K, vel_1d_ext, xsi_1d_ext = build_fusion_kernel(vel_1d, xsi_1d)
symv = sc.linalg.blas.get_blas_funcs('symv', (K,))
for i in range(len(zArr)):

    reactivity1D[i] = fusion_reactivity(zEvolF[i], vel_1d, xsi_1d, K, symv)

# Plot the data
fig = plt.figure(figsize=(12, 8), tight_layout=True)
ax = fig.add_subplot(111)

ax.plot(zArr, reactivity1D)
# Smooth it a little
smoothedReactivity1D = sc.signal.savgol_filter(reactivity1D, window_length=5, polyorder=3)
ax.plot(zArr, smoothedReactivity1D)

ax.set_title(r'10keV Maxwellian (n$_e$ = 5$\cdot$10$^{19}$m$^{-3}$)')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'Reactivity [#/(m$^3$s)]')

plt.show()

# =============================================
# Test with a fast ion + maxwellian
# =============================================

# Generate 10% fast ion plasma with a 1keV maxwellian
vel, xsi, f = dist_func(n_fast = 5e18,
                        n_max = 5e19,
                        T_max = 1,
                        R_m = 57,
                        E_NBI = 25,
                        theta_NBI = np.pi/4,
                        T_e = 0.075,
                        mu_i = 2,
                        Z_eff = 3,
                        gridsize = 50)

vel_1d = vel[0, :]
xsi_1d = xsi[:, 0]

# Evolve f along z. Reuses the K/symv built above since E_NBI, mu_i and
# gridsize (and therefore the vel/xsi grid) are unchanged from the previous
# section.
rAlongTube_all, bNormArr_all, RmArr = build_flux_tube_geometry(np.array([rMid]), zArr, Rmesh, Zmesh, Bmag, magneticFlux)
zEvolF = dist_func_z_evol(f, vel, xsi, bNormArr_all[0])

reactivity1D = np.zeros(shape=len(zArr))

# Calculate the reactivity at each z position
for i in range(len(zArr)):

    reactivity1D[i] = fusion_reactivity(zEvolF[i], vel_1d, xsi_1d, K, symv)

# Plot the data
fig = plt.figure(figsize=(12, 8), tight_layout=True)
ax = fig.add_subplot(111)

ax.plot(zArr, reactivity1D)
# Smooth it a little
smoothedReactivity1D = sc.signal.savgol_filter(reactivity1D, window_length=5, polyorder=3)
ax.plot(zArr, smoothedReactivity1D)

ax.set_title(r'1keV Maxwellian (n$_e$ = 5$\cdot$10$^{19}$m$^{-3}$) with 10% fast ions')
ax.set_xlabel('Z [m]')
ax.set_ylabel(r'Reactivity [#/(m$^3$s)]')

plt.show()
