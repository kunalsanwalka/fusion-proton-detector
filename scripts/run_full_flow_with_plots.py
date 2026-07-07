# -*- coding: utf-8 -*-
"""
This script runs the full flow of the synthetic proton detector simulation and makes plots along the way.

It runs and plots the following steps:

1. Midplane distribution function
2. Z-evolved distribution function
3. 2D reactivity profile
4. Detector particle tracks
5. Volume weights for a detector
6. Detector response for different reactivity profiles
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import *
from detector_instrument_function import *

global filenameEqdsk
filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

def plot_distribution_midplane():

    vel, xsi, f = dist_func(n_fast = 5e18,
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
    
    return vel, xsi, f

def plot_distribution_z_evolved(filenameEqdsk):

    # Distribution function at the midplane
    vel, xsi, f = dist_func(n_fast = 5e18,
                            n_max = 5e19,
                            T_max = 0.03,
                            R_m = 57,
                            E_NBI = 25,
                            theta_NBI = np.pi/4,
                            T_e = 0.075,
                            mu_i = 2,
                            Z_eff = 3,
                            gridsize = 500)

    # Load the magnetic equilibrium
    Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk)

    # Radial distance where the distribution function is calculated from the midplane
    rDist = 0.01 # [m]

    # z-positions where we want the evolved distribution function
    zValArr = np.linspace(0, 0.8, 20)

    # Precompute the flux tube geometry (B/B0 along the tube) for this single
    # starting radius, then evolve f along z with mu-conservation
    rAlongTube_all, bNormArr_all, RmArr = build_flux_tube_geometry(np.array([rDist]), zValArr, Rmesh, Zmesh, Bmag, magneticFlux)
    zEvolF = dist_func_z_evol(f, vel, xsi, bNormArr_all[0], makeplot=True)

def plot_reactivity_2D(filenameEqdsk):

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

    # Plot the radial density and temperature profiles
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(rArr*1e2, nArr/1e19, linewidth=3, label=r'$n_e$ [m$^{-3}$]')
    ax.plot(rArr*1e2, TeArr/10, linewidth=3, label=r'$T_e$ [10eV]')

    ax.set_xlabel('r [cm]')
    ax.set_ylabel('Density / Temperature')
    ax.set_title('Radial Profiles')
    
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    ax.legend()
    plt.show()

    # Calculate the (r,z) evolved distribution functions
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
    
    # Calculate the 2D reactivity profile
    zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(vel, xsi, zArr2D, rArr2D, f_rz,
                                                        makeplot=True, savename='full_workflow_reactivity.npz')
    
    return zArr2D, rArr2D, reactivity2D

def plot_detector_tracks(filenameEqdsk):

    b_field_interpolation(filenameEqdsk)

    detPos = np.array([-0.257,  0.307,  0.3])

    detPhi = (np.pi/180) * (290)

    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550

    bendRad = 0.7

    tubeAng = 10 * np.pi/180

    openingTracks = generate_tracks_aperture(detPos, detPhi, detSize, bendRad, tubeAng, makeplot=True)

def plot_volume_weights(filenameEqdsk):

    b_field_interpolation(filenameEqdsk)

    detPos = np.array([-0.257,  0.307,  0.3])

    detPhi = (np.pi/180) * (290)

    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550

    bendRad = 0.7

    tubeAng = 10 * np.pi/180

    _, _, _, _, _ = volume_weights(detPos, detPhi, detSize, bendRad, tubeAng,
                                   cellSize=1e-2, errorLim=None, maxParticles=5000,
                                   makeplot=True)

def plot_detector_response(filenameEqdsk, filenameReactivity):

    file = np.load('/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing.npz')

    # (x,y,z) positions. These are mounted on the CC inner wall.
    zPosArr = file['zPosArr'] # meters
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

    # Detector angles (after optimization)
    detPhiArr = file['optimalAngleArr'] * (np.pi/180) # radians

    # Detector sizes
    detRad = 1.0 # inches
    detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

    # Tube bend radii
    bendRadArr = np.full(len(zPosArr), 1.2) # meters

    # Tube sector angles
    tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians

    detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                            makeplot=True)

if __name__ == '__main__':

    # Step 1: Midplane distribution function
    # vel, xsi, f = plot_distribution_midplane()

    # Step 2: Z-evolved distribution function
    # plot_distribution_z_evolved(filenameEqdsk)

    # Step 3: 2D reactivity profile
    zArr2D, rArr2D, reactivity2D = plot_reactivity_2D(filenameEqdsk)

    # Step 4: Detector particle tracks
    # plot_detector_tracks(filenameEqdsk)

    # Step 5: Volume weights for a detector
    # plot_volume_weights(filenameEqdsk)

    # Step 6: Detector response for different reactivity profiles
    plot_detector_response(filenameEqdsk, 'full_workflow_reactivity.npz')