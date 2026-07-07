# -*- coding: utf-8 -*-
"""
Driver script for fusion_reactivity.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys

import numpy as np
import scipy as sc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import dist_func, extend_f, fusion_cross_section, fusion_reactivity, build_fusion_kernel

# %% Calculate reactivity for a single distribution function at the midplane
# Test to see if the fusion reactivity calculator is working properly

# =============================================
# Test with a maxwellian
# =============================================

# Generate a maxwellian with 10keV
vel, xsi, f = dist_func(n_fast = 5,
                        n_max = 5e19,
                        T_max = 10,
                        R_m = 57,
                        E_NBI = 250,
                        theta_NBI = np.pi/4,
                        T_e = 0.075,
                        mu_i = 2,
                        Z_eff = 3,
                        gridsize = 50,
                        makeplot = True)

vel_1d = vel[0, :]
xsi_1d = xsi[:, 0]

# Extend the distribution function
_, _, _ = extend_f(f, vel_1d, xsi_1d, makeplot=True)

fusion_cross_section()
K, vel_1d_ext, xsi_1d_ext = build_fusion_kernel(vel_1d, xsi_1d)
symv = sc.linalg.blas.get_blas_funcs('symv', (K,))
reactivity = fusion_reactivity(f, vel_1d, xsi_1d, K, symv)

print(reactivity/1e13)
