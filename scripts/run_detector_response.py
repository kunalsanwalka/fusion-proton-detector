# -*- coding: utf-8 -*-
"""
Driver script for generate_detector_response.

This function can be used to generate, plot and save the detector response for a given-
1. Magnetic equilibrium
2. Fusion reactivity profile
3. Detector geometry

It can also be used to compare the detector response for different fusion reactivity profiles.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector_instrument_function import generate_detector_response, load_detector_geometry

global plotDir
plotDir = '/home/sanwalka/synthetic_proton_detector/plots/'

# Detector geometry (positions, angles, sizes, bend radii and tube angles) saved by run_angle_optimization.py
geometryFile = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_whamr_cdr.npz'

def detector_geometry():
    """
    Load the detector geometry (positions, angles, sizes, bend radii and tube angles)
    """

    return load_detector_geometry(geometryFile)

def single_reactivity_response(filenameEqdsk, filenameReactivity,
                               makeplot=False, savename=''):
    """
    Generate the detector response for a single reactivity profile.
    """

    # Get the detector geometry
    detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr = detector_geometry()

    # Generate the detector response
    detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                             makeplot=makeplot, savename=savename)
    
    return

def compare_reactivity_responses(filenameEqdsk, filenameReactivityList, labelList,
                                 makeplot=False, savename=''):
    """
    Compare the detector response for different reactivity profiles.
    """

    # Get the detector geometry
    detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr = detector_geometry()

    # Generate the detector responses for each reactivity profile
    detResponseList = []
    for filenameReactivity in filenameReactivityList:
        detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                 makeplot=False)
        detResponseList.append(detResponse)

    # Plot the responses together
    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        for i, detResponse in enumerate(detResponseList):
            ax.plot(detPosArr[:, 2], detResponse/1e3, linewidth=3, label=labelList[i])
            ax.scatter(detPosArr[:, 2], detResponse/1e3, s=200)

        ax.legend()
        ax.set_ylim(0, None)
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('Response [counts/ms]')

        if savename:
            plt.savefig(plotDir+savename)

        plt.show()

    return

if __name__ == '__main__':

    #### Generate the detector response for a single reactivity profile
    
    # 2D reactivity profile
    filenameReactivity = 'predicted_reactivity_whamr_cdr.npz'

    # eqdsk file
    filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    filenameEqdsk = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/eq__pleiades_3/eqdsk'

    single_reactivity_response(filenameEqdsk, filenameReactivity, makeplot=True, savename='detector_response_whamr_cdr.npz')

    #### Compare the detector response for different reactivity profiles

    # # 2D reactivity profiles
    # filenameReactivityList = [
    #     'predicted_reactivity_only_fast_ions.npz',
    #     'predicted_reactivity_only_maxwell_ions.npz',
    #     'predicted_reactivity_10percent_fast_ions.npz'
    # ]

    # labelList = [
    #     'Fast only',
    #     'Maxwellian only',
    #     'Fast + Maxwellian'
    # ]

    # compare_reactivity_responses(filenameEqdsk, filenameReactivityList, labelList,
    #                              makeplot=True, savename='detector_response_comparison.png')