import os
import sys
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import *
from detector_instrument_function import generate_detector_response, load_detector_geometry

def calculate_2d_reactivity(filenameCQL3D, filenameEQDSK, savenameReactivity):
    """
    Calculate the 2D fusion reactivity profile from a CQL3D simulation.
    """

    # Name the plot after the CQL3D run (the directory above 'simulation_results')
    pathParts = os.path.normpath(filenameCQL3D).split(os.sep)
    runName = pathParts[pathParts.index('simulation_results') - 1] if 'simulation_results' in pathParts else 'cql3d'

    startTime = time.time()
    vel, xsi, zArr2D, rArr2D, f_rz = cql3d_dist_func_rz(filenameCQL3D, filenameEQDSK)
    distFuncTime = time.time()
    print(f'Time taken to generate f_rz = {np.round(distFuncTime - startTime, 2)}s')

    # Get the r-z evolved fusion reactivity profile
    zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(vel, xsi, zArr2D, rArr2D, f_rz,
                                                        makeplot=True,
                                                        savefigname=f'{runName}_2d_reactivity.png',
                                                        savename=savenameReactivity)
    reactivityTime = time.time()
    print(f'Time taken to generate the reactivity = {np.round(reactivityTime - distFuncTime, 2)}s')

    return

def run_full_flow(filenameCQL3D, filenameEQDSK, geometryFile, filenameReactivity=None, makeplot=False, savename=None):
    """
    Generate the detector response for a single reactivity profile.
    """

    # Get the detector geometry
    detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr = load_detector_geometry(geometryFile)

    # Generate the detector response
    # The plot is made below instead, alongside the density and reactivity
    detResponse = generate_detector_response(filenameEQDSK, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                             makeplot=False, savename=savename)

    if makeplot:

        #### Density along the innermost flux tube

        ds = xr.open_dataset(filenameCQL3D,
                             decode_timedelta=False)

        # Density [Time x R x Z x Species]
        # [cm^-3] to [m^-3]
        dens = ds['densz1'].values * 1e6
        # 1st one is ions, the 2nd is electrons. Pick the last timestep and the innermost flux tube.
        densZ = dens[-1, 0, :, 0]

        # z values of the innermost flux tube
        # [cm] to [m]
        zDens = ds['solzz'].values[0] * 1e-2

        #### Reactivity along r=0

        # The innermost flux tube is the closest point to r=0
        dataObj = np.load(reactivityDir+filenameReactivity)
        zReac = dataObj['zArr2D'][0]
        reactivityZ = dataObj['reactivity2D'][0]

        #### Plot

        # The units of each quantity are picked so that they all fit on the same y-axis
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        # Density
        ax.plot(zDens, densZ/1e19, linewidth=3,
                label=r'Density [10$^{19}$ m$^{-3}$]')

        # Reactivity
        ax.plot(zReac, reactivityZ/1e12, linewidth=3,
                label=r'Reactivity [10$^{12}$ #/m$^3$s]')

        # Detector response
        # [counts/s] to [10^5 counts/s]
        detLine, = ax.plot(detPosArr[:, 2], detResponse/1e5, linewidth=3,
                           label=r'Response [10$^{5}$ #/s]')
        ax.scatter(detPosArr[:, 2], detResponse/1e5, s=200, color=detLine.get_color())

        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, None)
        ax.set_xlabel('Z [m]')

        ax.legend()

        plt.savefig('/home/sanwalka/synthetic_proton_detector/plots/detector_response_density_reactivity.png',
                    dpi=300)

        plt.show()

    return detResponse

if __name__ == '__main__':
    
    # Filepath for the test CQL3D simulation
    filenameCQL3D = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/fp__cql3dm_4/WHAM.nc'

    # Filepath for the associated eqdsk file
    filenameEQDSK = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

    # Location to store the 2D fusion reactivity profile
    savenameReactivity = 'predicted_reactivity_cql3d_homemade.npz'

    # FIlepath for the detector geometry file
    geometryFile = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing_whamr.npz'

    calculate_2d_reactivity(filenameCQL3D, filenameEQDSK, savenameReactivity)

    # Generate the detector response
    detResponse = run_full_flow(filenameCQL3D, filenameEQDSK, geometryFile, 
                                filenameReactivity=savenameReactivity, 
                                makeplot=True)