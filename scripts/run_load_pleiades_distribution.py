# -*- coding: utf-8 -*-
"""
Driver script for dist_func_rz_pleiades.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter) - run the
one you need instead of editing predicted_fusion_reactivity.py.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import dist_func_rz_pleiades

# %% Load pleiades output
# Load the distribution function from a pleaides output.

# Pleaides output file location
filenamePleiades = '/home/sanwalka/synthetic_proton_detector/data/pleiades_260105053.h5'

# eqdsk output file location
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

vPar, vPerp, zArr2D, rArr2D, f_rz = dist_func_rz_pleiades(filenamePleiades, filenameEqdsk)
