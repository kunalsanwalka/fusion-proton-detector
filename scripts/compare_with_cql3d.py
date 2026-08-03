"""
Driver script to test pulling the distribution function data from a CQL3D simulation
"""

import os
import sys
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import *

if __name__ == '__main__':
    
    # Filepath for the test CQL3D simulation
    filenameCQL3D = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/fp__cql3dm_4/WHAM.nc'
    ds = xr.open_dataset(filenameCQL3D,
                         decode_timedelta=False)

    # Filepath for the associated eqdsk file
    filenameEQDSK = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

    # Location to store the 2D fusion reactivity profile
    savenameReactivity = 'predicted_reactivity_cql3d_homemade.npz'

    # Load the midplane distribution function from the CQL3D simulation
    # Y, X, rArr, f = load_cql3d_f(filenameCQL3D, makeplot=True, logplot=True)
    
    startTime = time.time()
    vel, xsi, zArr2D, rArr2D, f_rz = cql3d_dist_func_rz(filenameCQL3D, filenameEQDSK)
    distFuncTime = time.time()
    print(f'Time taken to generate f_rz = {np.round(distFuncTime - startTime, 2)}s')

    # Get the r-z evolved fusion reactivity profile
    zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(vel, xsi, zArr2D, rArr2D, f_rz,
                                                        makeplot=True,
                                                        savename=savenameReactivity)
    reactivityTime = time.time()
    print(f'Time taken to generate the reactivity = {np.round(reactivityTime - distFuncTime, 2)}s')

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.scatter(zArr2D.flatten(), rArr2D.flatten())
    plt.show()

    # Total fusion power for 4 different reactions
    fuspwrvt = ds['fuspwrvt'].values # W
    ddPower = fuspwrvt[2] + fuspwrvt[3]

    # Energy per reaction for D + D --> n + 3He and D + D --> p + T
    nHeEnergy = 2.45 + 0.82 # MeV
    pTEnergy = 1.01 + 3.02 # MeV
    # Convert to Joules
    nHeEnergy *= 1e6 * sc.constants.e # J
    pTEnergy *= 1e6 * sc.constants.e # J

    # Reaction rates for D + D --> n + 3He and D + D --> p + T
    # fuspwrvt indexing (verified against the CQL3D source, sigsetup.f):
    # 0 = D+T->a+n, 1 = D+He3->a+p, 2 = D+D->T+p, 3 = D+D->He3+n.
    # This is the reverse of what was previously assumed here.
    nHeRate = fuspwrvt[3] / nHeEnergy # reactions/s
    pTRate = fuspwrvt[2] / pTEnergy # reactions/s
    
    print(f'D + D --> p + T reaction rate = {pTRate:.2e} reactions/s')