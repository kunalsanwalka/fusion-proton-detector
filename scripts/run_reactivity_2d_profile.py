# -*- coding: utf-8 -*-
"""
Driver script for dist_func_rz / fusion_reactivity_rz.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import dist_func_rz, fusion_reactivity_rz

# %% Calculate 2D reactivity profile

# Location of magnetic equilibrium file
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

# Location to store the 2D fusion reactivity profile
savenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_only_maxwell_ions.npz'

# Get the r-z evolved distribution functions
rArr = np.linspace(0, 0.1, 10)
zArr = np.linspace(0, 0.8, 50)

# Use density and temperature profiles from Keisuke's paper.
nArr = 5e19 - 3e20*rArr
TeArr = 75 - (80*rArr)**2

# Fast ion component is 1/10 the maxwellian background
nFastArr = nArr/10
nMaxArr = nArr - nFastArr

# Assume the maxwellian background has a temperature of 30eV
TMaxArr = np.full(np.shape(rArr), 0.03)

# Zeff is flat at 2 for simplicity
ZeffArr = np.full(np.shape(rArr), 2)

# NBI parameters are the same for all flux surfaces
E_NBI = 25
theta_NBI = np.pi/4
mu_i = 2

startTime = time.time()
vel, xsi, zArr2D, rArr2D, f_rz = dist_func_rz(rArr,
                                              zArr,
                                              nFastArr,
                                              nMaxArr,
                                              TMaxArr,
                                              TeArr,
                                              ZeffArr,
                                              E_NBI,
                                              theta_NBI,
                                              mu_i,
                                              filenameEqdsk)
distFuncTime = time.time()
print(f'Time taken to generate f_rz = {np.round(distFuncTime - startTime, 2)}s')

# Get the r-z evolved fusion reactivity profile
zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(vel, xsi, zArr2D, rArr2D, f_rz,
                                                    makeplot=False,
                                                    savename=savenameReactivity)
reactivityTime = time.time()
print(f'Time taken to generate the reactivity = {np.round(reactivityTime - distFuncTime, 2)}s')
