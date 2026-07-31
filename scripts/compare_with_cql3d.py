"""
Driver script to test pulling the distribution function data from a CQL3D simulation
"""

import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import load_cql3d_f, magnetic_equilibrium

# Filepath for the test CQL3D simulation
filenameCQL3D = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/fp__cql3dm_4/WHAM.nc'

# Filepath for the associated eqdsk file
filenameEQDSK = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

# vel, xsi, rArr, f = load_cql3d_f(filenameCQL3D, makeplot=True, logplot=True)

Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEQDSK, makeplot=True)