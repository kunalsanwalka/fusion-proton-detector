# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:16:51 2026

@author: kunal

This program creates a parameterized distribution function from Sam's
implementation of Jan's formula and then does the following-
1. Evolve it along z for a given magnetic equilibrium
2. Calculate the fusion reactivity profile along r and z

"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# print('predicted_fusion_reactivity.py')
# print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
# print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
# print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))

import xarray as xr
import subprocess
import time
import h5py
import skimage
from egedal_f_obj import *
from itertools import chain
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from multiprocessing import Pool, cpu_count
from functools import partial

# Use TkAgg for interactive plotting
plt.switch_backend('TkAgg')

# Change the font size
plt.rcParams.update({'font.size': 26})

global plotDir
plotDir = '/home/sanwalka/synthetic_proton_detector/plots/'

def dist_func(n_fast, n_max, T_max, R_m, E_NBI, theta_NBI, T_e, mu_i, Z_eff, gridsize=100, 
              makeplot=False, savename=None):
    """
    This function calculates the distribution function at the midplane for a
    given set of plasma parameters

    Parameters
    ----------
    n_fast : float
        Fast ion plasma density (used for normalization). [particles / m^-3]
    n_max : float
        Maxwellian background plasma density (used for normalization). [particles / m^-3]
    T_max : float
        Maxwellian background plasma temperature. [keV]
    R_m : float
        Mirror ratio.
    E_NBI : float
        NBI injection energy. [keV]
    theta_NBI : float
        Injection angle for the NBI. [radians]
    T_e : float
        Electron temperature. [keV]
    mu_i : int
        Mass of ion species. [amu]
        Hydrogen = 1; Deuterium = 2
    Z_eff : float
        Effective charge of ions. [float]
    gridsize : int
        Size of the velocity grid in 1 dimension
    makeplot : boolean, optional
        Make a plot of the distribution function. 
        The default is False.
    savename : string, optional
        Name of the file to save the plot as. 
        The default is None.

    Returns
    -------
    vPar : np.array
        v_|| array.
    vPerp : np.array
        v_perp array.
    f : np.array
        Density normalized distribution function.
    """
    
    # Constants
    q = sc.constants.e
    m_p = sc.constants.m_p
    
    # ======================================
    # Fast ion component via egedal
    # ======================================

    # Injection velocity of the NBI [m/s]
    vNBI = (2*1e3*E_NBI*q/(m_p*mu_i))**0.5
    
    # Create the distribution function object
    fObj = egedal_f(R_m = R_m,
                    E_NBI = E_NBI,
                    theta_NBI = theta_NBI,
                    T_e = T_e,
                    mu_i = mu_i,
                    Z_eff = Z_eff,
                    N_j=10)
        
    # Define the velocity grids
    v = np.linspace(0, vNBI, gridsize)
    xsi = np.linspace(0, 1, gridsize) # normalized pitch angle
    X, Y = np.meshgrid(xsi, v)
    
    # Obtain the distribution function
    f = fObj.f(Y, X)
    
    # Convert the velocity grids to vPar and vPerp
    theta = np.arccos(X)
    vPar = Y*np.cos(theta)
    vPerp = Y*np.sin(theta)
    
    # Make a uniform grid in vPar and vPerp
    vPAR, vPERP = np.meshgrid(np.linspace(0, vNBI, gridsize), np.linspace(0, vNBI, gridsize))
    
    # Put the distribution function on the uniform grid
    f1D = f.flatten()
    vPerp1D = vPerp.flatten()
    vPar1D = vPar.flatten()
    velPoints = np.array([vPar1D, vPerp1D]).T
    
    fOnGrid = sc.interpolate.griddata(velPoints, f1D, (vPAR, vPERP), method='cubic', fill_value=0)
    
    # Rename stuff
    fFast = fOnGrid
    vPar = vPAR
    vPerp = vPERP

    # Normalize the fast ion distribution function to n_fast
    integrand = 2 * np.pi * vPerp * fFast # Cylindrical coordinates jacobian
    fastDens = np.trapezoid(np.trapezoid(integrand, vPerp[:, 0], axis=0), vPar[0], axis=0)
    fFast *= n_fast/fastDens

    # ======================================
    # Maxwellian background component
    # ======================================

    vTot = np.sqrt(vPar**2 + vPerp**2)

    tempInJoules = T_max*1e3*q
    ionMass = m_p*mu_i

    fMaxwell = np.exp(-ionMass*vTot**2/(2*tempInJoules))

    # Normalize the Maxwellian distribution function to n_max
    integrand = 2 * np.pi * vPerp * fMaxwell # Cylindrical coordinates jacobian
    maxwellDens = np.trapezoid(np.trapezoid(integrand, vPerp[:, 0], axis=0), vPar[0], axis=0)
    fMaxwell *= n_max/maxwellDens

    # ======================================    
    # Combined distribution function
    # ======================================

    f = fFast + fMaxwell
    f = np.clip(f, a_min=0, a_max=None)

    if makeplot == True:
        
        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Normalize first, then clip to log-safe minimum BEFORE passing to contourf
        f_norm = f / np.max(f)
        vmin_log = 1e-4
        f_norm = np.clip(f_norm, a_min=vmin_log, a_max=1)  # kill exact zeros
        
        levels = np.logspace(np.log10(vmin_log), 0, 100)   # log-spaced levels
        
        pltObj = ax.contourf(vPar/vNBI, vPerp/vNBI, f_norm,
                             norm=colors.LogNorm(vmin=vmin_log, vmax=1),
                             levels=levels,
                             extend='neither')
        
        cbar = fig.colorbar(pltObj, ax=ax)
        cbar.set_label(r'f/f$_{max}$')
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        
        # Add a text box with the plasma parameters
        textstr = '\n'.join((
            r'$E_{NBI}$ = '+str(E_NBI)+r' keV',
            r'$T_e$ = '+str(int(T_e*1e3))+r' eV',
            r'$Z_{eff}$ = '+str(Z_eff),
            r'$R_m$ = '+str(R_m)))
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=22,
                horizontalalignment='left', verticalalignment='top', bbox=props)
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        ax.set_title(r'$f_{z=0}(v_{||}, v_{\perp})$')
        
        if savename is not None:
            plt.savefig(plotDir+savename, dpi=300)

        plt.show()
        
    return vPar, vPerp, f

def magnetic_equilibrium(filenameEqdsk, makeplot=False):
    """
    This function creates interpolation functions for the magnetic fields for a
    given eqdsk file.
    
    To get the data from any of the interpolation functions, they can be called
    like-
    interpFunc([zPos, rPos])[0]

    Because of shared environment nonsense, this script launches a daughter 
    script in a different virtual environment which stores the data in an .npz
    file and then that file is read back by this script.

    Parameters
    ----------
    filenameEqdsk : string
        Location of the eqdsk.
    makeplot : boolean, optional
        Make a plot of the magnetic fields. The default is False.

    Returns
    -------
    Rmesh : np.array
        2D array of radial positions
    Zmesh : np.array
        2D array of axial positions
    Br : np.array
        Br values. [Tesla]
    Bz : np.array
        Bz values. [Tesla]
    Bmag : np.array
        |B| values. [Tesla]
    magneticFlux : np.array
        Magnetic flux. [Weber]
    """

    # Start the script to create an .npz file with the quantities of interest
    # in the shared pleiades_env
    subprocess.run(['/share/envs/pleiades_env/bin/python',
                    'eqdsk_analysis_functions.py',
                    '-filename', filenameEqdsk], check=True)

    # Load the data from the .npz file made by this function
    npzFilename = np.load('filenameEqdsk'+'.npz')

    Rmesh = npzFilename['Rmesh']
    Zmesh = npzFilename['Zmesh']
    Br = npzFilename['Br']
    Bz = npzFilename['Bz']
    Bmag = npzFilename['Bmag']
    magneticFlux = npzFilename['magneticFlux']
        
    if makeplot == True:
        
        #### Plot the magnetic fields
        fig = plt.figure(figsize=(15, 21), tight_layout=True)
        
        # Br
        ax = fig.add_subplot(311)
        
        pltObj = ax.contourf(Zmesh, Rmesh, Br, levels=100, cmap='inferno')
        pltObj = ax.contourf(Zmesh, -Rmesh, Br, levels=100, cmap='inferno')
        
        cmap = fig.colorbar(pltObj, ticks=np.arange(np.round(np.min(Br)), np.round(np.max(Br)), 2), label='[Tesla]')

        ax.set_title(r'$B_r$')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        
        # Bz
        ax = fig.add_subplot(312)
        
        pltObj = ax.contourf(Zmesh, Rmesh, Bz, levels=100, cmap='inferno')
        pltObj = ax.contourf(Zmesh, -Rmesh, Bz, levels=100, cmap='inferno')
        
        cmap = fig.colorbar(pltObj, ticks=np.arange(np.round(np.min(Bz)), np.round(np.max(Bz)), 2), label='[Tesla]')

        ax.set_title(r'$B_z$')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        
        # |B|
        ax = fig.add_subplot(313)
        
        pltObj = ax.contourf(Zmesh, Rmesh, Bmag, levels=100, cmap='inferno')
        pltObj = ax.contourf(Zmesh, -Rmesh, Bmag, levels=100, cmap='inferno')
        
        cmap = fig.colorbar(pltObj, ticks=np.arange(np.round(np.min(Bmag)), np.round(np.max(Bmag)), 2), label='[Tesla]')

        ax.set_title(r'$|\vec{B}|$')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        
        plt.show()
        
        #### Plot the fluxes
        fig = plt.figure(figsize=(15, 7), tight_layout=True)
        ax = fig.add_subplot(111)
        
        pltObj = ax.contour(Zmesh, Rmesh, magneticFlux, levels=np.linspace(1e-3, 0.05, 25), cmap='inferno')
        pltObj = ax.contour(Zmesh, -Rmesh, magneticFlux, levels=np.linspace(1e-3, 0.05, 25), cmap='inferno')
        
        cmap = fig.colorbar(pltObj, ticks=np.arange(np.round(np.min(magneticFlux), 1), np.round(np.max(magneticFlux), 1)+0.01, 0.1), label='[Webers]')
        
        ax.set_title(r'Magnetic Flux')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_aspect('equal')
        
        plt.show()
    
    return Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux

def dist_func_z_evol(f, vPar, vPerp, rDist, zValArr, Rmesh, Zmesh, Bmag, magneticFlux, makeplot=False):
    """
    This function evolves the distribution along z for a given magnetic
    equilibrium. It does this via the mu (1st adiabatic invariant) 
    conservation.

    Parameters
    ----------
    f : np.array
        2D distribution function.
    vPar : np.array
        v_|| array. [m/s OR normalized]
        Units can be m/s OR normalized. As long as vPar and vPerp units match.
    vPerp : np.array
        v_perp array. [m/s OR normalized]
        Units can be m/s OR normalized. As long as vPar and vPerp units match.
    rDist : float
        Radial distance where this distribution function is calculated from the
        midplane. [m]
    zValArr : np.array
        1D array that contains the z-positions where we want the evolved
        distribution function.
    Rmesh : np.array
        2D array of radial positions
    Zmesh : np.array
        2D array of axial positions
    Bmag : np.array
        |B| values. [Tesla]
    magneticFlux : np.array
        Magnetic flux. [Weber]
    makeplot : boolean, optional
        Make a plot of the distribution function at 2 z-positions to make sure
        the function is working properly. The default is False.

    Returns
    -------
    zEvolf : np.array
        3D array with the z-evolved distribution function.
    rValArr : np.array
        r-values that map along the flux tube specified by rDist.
    """
    
    # ==================================================================
    # Magnetic field strength along the flux tube
    # ==================================================================

    # Get the magnetic flux for the distribution function at the midplane
    idx = np.abs(Rmesh[0] - rDist).argmin()
    midplaneFlux = magneticFlux[int(len(Zmesh)/2), idx]

    contours = skimage.measure.find_contours(magneticFlux, level=midplaneFlux)

    if len(contours) > 0:

        contour = contours[0]  # shape: (N, 2), columns are [row_idx, col_idx]

        # find_contours returns fractional indices — interpolate into the real
        # coordinate arrays rather than rounding to the nearest grid point
        z_axis = Zmesh[:, 0]  # 1D array of Z values
        r_axis = Rmesh[0]     # 1D array of R values

        z_idx_interp = sc.interpolate.interp1d(np.arange(len(z_axis)), z_axis)
        r_idx_interp = sc.interpolate.interp1d(np.arange(len(r_axis)), r_axis)

        z_vals = z_idx_interp(contour[:, 0])
        r_vals = r_idx_interp(contour[:, 1])

        contour_interp = sc.interpolate.interp1d(z_vals, r_vals,
                                                bounds_error=False, fill_value=np.nan)
        rValArr = contour_interp(zValArr)
        
    # Make an interpolation function for |B|
    BmagInterpolator = sc.interpolate.RegularGridInterpolator((Zmesh[:,0], Rmesh[0]), Bmag)
    
    # Get the magnetic field strength data for each (z, r) pair
    points = np.array([zValArr, rValArr]).T
    bMagArr = BmagInterpolator(points)
    
    # Magnetic field strength at z=0
    bNorm = BmagInterpolator([0, rDist])[0]
    
    # B/B_0
    bNormArr = bMagArr/bNorm

    # =======================================================================
    # Evolve the distribution function along z due to mu and E conservation
    # =======================================================================

    zEvolf = np.zeros(shape=(len(zValArr), *f.shape))
    
    # Precompute all transformed grids at once (vectorized over z)
    b = bNormArr[:, None, None]          # shape (Nz, 1, 1)
    vPerpNew = vPerp[None] * np.sqrt(b)  # shape (Nz, Nperp, Npar)
    vParNew  = np.sqrt(vPar[None]**2 + vPerp[None]**2 * (1 - b))

    # Smooth the data BEFORE interpolation. This helps knock down some of the spikes
    fSmooth = sc.ndimage.gaussian_filter(f, sigma=2)

    # Flatted f (used for interpolation)
    distDataFlat = fSmooth.ravel()

    for i in range(len(zValArr)):
    
        # Get the velocity grids at the current z position
        vPerpFlat = vPerpNew[i].ravel()
        vParFlat  = vParNew[i].ravel()

        # Ignore points where vPar is NaN (trapped particles that don't make it out as far)
        valid  = ~np.isnan(vParFlat)
        points = np.column_stack((vParFlat[valid], vPerpFlat[valid]))
        values = distDataFlat[valid]

        # Triangulate the data
        tri = sc.spatial.Delaunay(points)
        # Build the interpolator
        interp = sc.interpolate.CloughTocher2DInterpolator(tri, values, fill_value=0.0)

        # Extract the result
        result = interp(vPar, vPerp)
        
        # Smooth the output to knock down spikes again
        result = sc.signal.savgol_filter(result, window_length=5, polyorder=2, axis=0)
        result = sc.signal.savgol_filter(result, window_length=5, polyorder=2, axis=1)

        # Clean up and add to zEvolf
        result[result < 0] = 0.0
        zEvolf[i] = result
    
    if makeplot == True:
        
        fig = plt.figure(figsize=(12, 16), tight_layout=True)

        #### Midplane plot
                
        ax = fig.add_subplot(211)
        
        # Normalize first, then clip to log-safe minimum BEFORE passing to contourf
        index = 0
        normVal = np.max(zEvolf[index])
        f_norm = zEvolf[index] / normVal
        vmin_log = 1e-4
        f_norm = np.clip(f_norm, a_min=vmin_log, a_max=1)  # kill exact zeros
        
        levels = np.logspace(np.log10(vmin_log), 0, 100)   # log-spaced levels

        pltObj = ax.contourf(vPar/np.max(vPar), vPerp/np.max(vPerp), f_norm, 
                             norm=colors.LogNorm(vmin=vmin_log, vmax=1),
                             levels=levels,
                             extend='neither')
        
        cbar = fig.colorbar(pltObj, ax=ax)
        cbar.set_label(r'f/f$_{max}$')
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        ax.set_title(f'z = {np.round(zValArr[index], 2)}m', 
                     pad=45)
        
        #### Off-midplane plot

        ax = fig.add_subplot(212)
        
        # Normalize first, then clip to log-safe minimum BEFORE passing to contourf
        index = -3
        f_norm = zEvolf[index] / normVal
        vmin_log = 1e-4
        f_norm = np.clip(f_norm, a_min=vmin_log, a_max=1)  # kill exact zeros
        
        levels = np.logspace(np.log10(vmin_log), 0, 100)   # log-spaced levels

        pltObj = ax.contourf(vPar/np.max(vPar), vPerp/np.max(vPerp), f_norm, 
                             norm=colors.LogNorm(vmin=vmin_log, vmax=1),
                             levels=levels,
                             extend='neither')
        
        cbar = fig.colorbar(pltObj, ax=ax)
        cbar.set_label(r'f/f$_{max}$')
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        cbar.set_ticklabels([r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        ax.set_title(f'z = {np.round(zValArr[index], 2)}m', 
                     pad=45)
        
        plt.show()
    
    return zEvolf, rValArr

def compute_row_fz(i, nFastArr, nMaxArr, TMaxArr, RmArr, E_NBI, theta_NBI, TeArr, mu_i, ZeffArr, gridsize, rArr, zArr, Rmesh, Zmesh, Bmag, magneticFlux):
    """
    Computes f along z for a given set of r and z values. This is purely a
    helper function that allows dist_func_rz to run things in
    parallel.
    """

    # Get the distribution function at the midplane
    vPar, vPerp, midplaneF = dist_func(nFastArr[i], nMaxArr[i], TMaxArr[i], RmArr[i], E_NBI, theta_NBI, TeArr[i], mu_i, ZeffArr[i], gridsize=gridsize)
    
    # Evolve it along zVals
    f_z, rAlongTube = dist_func_z_evol(midplaneF, vPar, vPerp, rArr[i], zArr, Rmesh, Zmesh, Bmag, magneticFlux)
    
    return i, f_z, rAlongTube, vPar, vPerp

def dist_func_rz(rArr, zArr, nFastArr, nMaxArr, TMaxArr, TeArr, ZeffArr, E_NBI, theta_NBI, mu_i, filenameEqdsk):
    """
    This function generates a distribution function at each (r,z) location in
    the plasma.
    
    It generates a unique distribution function at the midplane for each radial
    position as given in rArr and then evolves it along zArr with mu
    conservation.
    
    Since each distribution function at each flux surface is unique, they can
    all have different values of-
    1. Density (via nArr)
    2. T_e (via TeArr)
    3. Z_eff  (via ZeffArr)
    
    Some things do not change between flux tubes which are specified as floats
    such as-
    1. NBI Energy (via E_NBI)
    2. NBI injection angle (via theta_NBI)
    3. Ion species (via mu_i)

    Parameters
    ----------
    rArr : np.array
        Radial positions at the midplane where we want to find the distribution
        function. [meters]
    zArr : np.array
        Axial positions where want the distribution functions. [meters]
    nFastArr : np.array
        Fast ion component plasma density profile at the midplane. [m^-3]
    nMaxArr : np.array
        Maxwellian background plasma density profile at the midplane. [m^-3]
    TMaxArr : np.array
        Maxwellian background plasma temperature profile at the midplane. [keV]
    TeArr : np.array
        T_e profile at the midplane. [keV]
    ZeffArr : np.array
        Effective charge state profile at the midplane.
    E_NBI : float
        Injection energy of the NBI. [keV]
    theta_NBI : float
        Injection angle of the NBI. [radians]
    mu_i : int
        Mass of ion species. [amu]
        Hydrogen = 1; Deuterium = 2
     filenameEqdsk : string
         Location of the eqdsk.

    Returns
    -------
    vPar : np.array
        v_|| array. [m/s]
    vPerp : np.array
        v_perp array. [m/s]
    zArr2D : np.array
        z-locations where the f has been calculated. [meters]
    rArr2D : np.array
        r-locations where the f has been calculated. [meters].
    f_rz : np.array
        Distribution function at all points in the plasma.
        This is a 4D array with the following index convention-
        1st index - r-position
        2nd index - z-position
        3rd index - v_||
        4th index - v_perp
    """
    
    # =========================================================================
    # Calculate the mirror ratios for each flux surface as defined by rArr for
    # the input magnetic equilibrium.
    # =========================================================================
    
    # Load the magnetic equilibirum
    Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk,
                                                                    makeplot=False)
    
    # Make an interpolation function for |B| and Phi
    fluxInterpolator = sc.interpolate.RegularGridInterpolator((Zmesh[:,0], Rmesh[0]), magneticFlux)
    BmagInterpolator = sc.interpolate.RegularGridInterpolator((Zmesh[:,0], Rmesh[0]), Bmag)
    
    # Flux and |B| values that correspond to the rArr values
    midplaneFluxArr = np.zeros_like(rArr)
    midplaneBmagArr = np.zeros_like(rArr)
    for i in range(len(rArr)):
        midplaneFluxArr[i] = fluxInterpolator([0, rArr[i]])[0]
        midplaneBmagArr[i] = BmagInterpolator([0, rArr[i]])[0]
        
    # r-location at the mirror throat for the given flux surfaces
    rArrAtThroat = np.zeros_like(rArr)
    for i in range(len(rArr)):
        
        contour = plt.contour(Rmesh, Zmesh, magneticFlux, levels=[midplaneFluxArr[i]])
        plt.close()
        
        # Extract the contour path
        paths = contour.get_paths()
        if len(paths) > 0:
            vertices = paths[0].vertices  # [r, z] pairs along the contour
            
            # Interpolate to get r for each z in zValArr
            contour_interp = sc.interpolate.interp1d(vertices[:, 1], vertices[:, 0], 
                                                     bounds_error=False, fill_value=np.nan)
            # use 0.95x the maximum mesh value as sometimes the data can be
            # wonky at the end of the grids
            rArrAtThroat[i] = contour_interp(np.max(Zmesh)*0.95)
            
    # Calculate the mirror ratio for each flux surface
    RmArr = np.zeros_like(rArr)
    for i in range(len(rArr)):
        # use 0.95x the maximum mesh value as sometimes the data can be
        # wonky at the end of the grids
        bMagInThroat = BmagInterpolator([np.max(Zmesh)*0.95, rArrAtThroat[i]])[0]
        RmArr[i] = bMagInThroat/midplaneBmagArr[i]
       
    # =========================================================================
    # Generate the midplane distribution functions and then evolve them along
    # zArr to get a 2D map of the distribution functions
    # =========================================================================
    
    # Size of the velocity grid
    gridsize = 50

    # This will be a whopper array that will store all the distribution functions
    f_rz = np.zeros(shape=(len(rArr), len(zArr), gridsize, gridsize))
    vPar = None
    vPerp = None
    rArr2D = []
    
    # Setup so that compute_row can only be called with i. The rest of the arguments are pre-setup.

    # Change the plotting backend to be non-interactive for the worker function since it will be called in parallel and we don't want multiple plot windows popping up.
    plt.switch_backend('Agg')

    worker = partial(compute_row_fz, 
                     nFastArr=nFastArr,
                     nMaxArr=nMaxArr,
                     TMaxArr=TMaxArr,
                     RmArr=RmArr, 
                     E_NBI=E_NBI, 
                     theta_NBI=theta_NBI, 
                     TeArr=TeArr, 
                     mu_i=mu_i, 
                     ZeffArr=ZeffArr, 
                     gridsize=gridsize,
                     rArr=rArr,
                     zArr=zArr, 
                     Rmesh=Rmesh, 
                     Zmesh=Zmesh, 
                     Bmag=Bmag, 
                     magneticFlux=magneticFlux)

    # Use half of the cores on lana
    with Pool(int(cpu_count()/2)) as pool:
        results = pool.map(worker, range(len(rArr)))

        # Unpack results
        for i, f_z, rAlongTube, vPar, vPerp in results:
            f_rz[i] = f_z
            rArr2D.append(rAlongTube)
            vPar = vPar
            vPerp = vPerp

    # Change the plotting backend back to interactive for the rest of the code.
    plt.switch_backend('TkAgg')

    rArr2D = np.array(rArr2D)

    # for i in range(len(rArr)):
        
    #     # Get the distribution function at the midplane
    #     vPar, vPerp, midplaneF = dist_func(nArr[i], RmArr[i], E_NBI, theta_NBI, TeArr[i], mu_i, ZeffArr[i], 
    #                                        gridsize=gridsize, 
    #                                        makeplot=False)
        
    #     # Evolve it along zVals
    #     f_z, rAlongTube = dist_func_z_evol(midplaneF, vPar, vPerp, rArr[i], zArr, Rmesh, Zmesh, Bmag, magneticFlux)
        
    #     f_rz[i] = f_z
    #     rArr2D.append(rAlongTube)
        
    # # Convert to numpy
    # f_rz = np.array(f_rz)
    # rArr2D = np.array(rArr2D)
    
    # Make the z-array also 2D to make the analysis easier
    zArr2D = np.tile(zArr, (len(rArr), 1))
    
    return vPar, vPerp, zArr2D, rArr2D, f_rz

def dist_func_rz_cql3d(filenameCQL3D):

    vPar, vPerp, zArr2D, rArr2D, f_rz = [], [], [], [], []

    # Open the CQL3D output file
    # with xr.open_dataset(filenameCQL3D) as ds:

        

    return vPar, vPerp, zArr2D, rArr2D, f_rz

def dist_func_rz_pleiades(filenamePleiades, filenameEqdsk):

    # =========================================================================
    # Open the pleiades output file and extract the distribution function 
    # parameters.
    # =========================================================================

    with h5py.File(filenamePleiades, 'r') as f:

        print("Keys in the file:", list(f.keys()))

        # Open the equilibrium object
        equilObj = f['Equilibrium']
        # Open the currents object
        currentsObj = f['Currents']
        # Open the diagnostics object
        diagnosticsObj = f['Diagnostics']
        # Open the mesh object
        meshObj = f['Mesh']
        # Open the vacuum fields object
        vacuumFieldsObj = f['VacuumFields']
        # Open the flux gridded pressure object
        fluxGriddedPressureObj = f['FluxGriddedPressure']

        print("Keys in Equilibrium object:", list(equilObj.keys()))
        print("Keys in Currents object:", list(currentsObj.keys()))
        print("Keys in Diagnostics object:", list(diagnosticsObj.keys()))
        print("Keys in Mesh object:", list(meshObj.keys()))
        print("Keys in VacuumFields object:", list(vacuumFieldsObj.keys()))
        print("Keys in FluxGriddedPressure object:", list(fluxGriddedPressureObj.keys()))

    vPar, vPerp, zArr2D, rArr2D, f_rz = [], [], [], [], []

    return vPar, vPerp, zArr2D, rArr2D, f_rz

def fusion_cross_section(makeplot=False):
    """
    This function generates a global interpolation function for the D(d,p)T
    fusion cross section. Since this cross section is universal, it is declared
    as a global variable for easy use by other functions.
    
    The function ddptFusionCXFunc takes the center of mass energy in eV units
    and outputs the fusion cross section in m^2 units.

    Parameters
    ----------
    makeplot : boolean, optional
        Make a plot of the fusion cross section. The default is False.
    """
    
    # Fusion cross section function for D(d,p)T is universal so it can be made
    # a global variable
    global ddptFusionCXFunc
    
    # Code in this block is from https://scipython.com/blog/plotting-nuclear-fusion-cross-sections/
    
    # Cross sections data directory
    crossSectionsDir = '/home/sanwalka/synthetic_proton_detector/cross_sections/'
    
    # To plot using centre-of-mass energies instead of lab-fixed energies, set True
    COFM = True
    
    # Reactant masses in atomic mass units (u).
    masses = {'D': 2.014, 'T': 3.016, '3He': 3.016}
    
    # Energy grid [eV]
    energyGrid = np.arange(1, 1e6, 10000)
    energyGrid = np.logspace(1, 6, 10000)
    
    def read_xsec(filenameCX):
        """Read in cross section from filename and interpolate to energy grid."""
        
        # E has units of MeV and xs has units of barns
        E, xs = np.genfromtxt(filenameCX, comments='#', skip_footer=2, unpack=True)
        
        # Convert to eV and m^2
        E *= 1e6
        xs /= 1e28
        
        # Remove all the directory information from the filename
        filenameCX = filenameCX.split('/')[-1]
        
        if COFM:
            
            collider, target = filenameCX.split('_')[:2]
            m1, m2 = masses[target], masses[collider]
            E *= m1 / (m1 + m2)
    
        xs = np.interp(energyGrid, E, xs)
        return xs
    
    # D + D -> T + p
    DDa_xs = read_xsec(crossSectionsDir + 'D_D_-_T_p.txt')
    
    ddptFusionCXFunc = sc.interpolate.CubicSpline(energyGrid, DDa_xs)
    
    if makeplot == True:
        
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        ax.plot(energyGrid/1e3, ddptFusionCXFunc(energyGrid), linewidth=3)
        
        ax.set_xlabel('E (CoM) [keV]')
        ax.set_ylabel(r'$\sigma$ [m$^2$]')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1, 1e3)
        ax.set_ylim(1e-32, 1e-27)
        
        ax.set_title('D(d,p)T cross section')
        
        plt.show()
    
    return

def fusion_reactivity(f, vPar_1d, vPerp_1d):
    """
    Calculate fusion reactivity for arbitrary 2D distribution function.
    
    Parameters:
    -----------
    f : 2D numpy array
        Distribution function f(v_||, v_perp) in s^3/m^6
        Shape should match (len(vPar), len(vPerp))
    vPar_1d : 1D numpy array  
        Parallel velocity grid in m/s
    vPerp_qd : 1D numpy array
        Perpendicular velocity grid in m/s
    
    Returns:
    --------
    reactivity : float
        Fusion reactivity <sigma*v> in m^3/s
    """
   
    # Reduced mass for identical particles [kg]
    m_reduced = sc.constants.m_p / 2.0
    
    n_par = len(vPar_1d)
    n_perp = len(vPerp_1d)
    
    # print(f"Starting vectorized 4D integration over {n_perp}x{n_par}x{n_perp}x{n_par} = {n_perp*n_par*n_perp*n_par:,} points...")
    
    # Create 4D grids using broadcasting
    # Dimensions: [v1_perp, v1_par, v2_perp, v2_par]
    # Shape will be: (n_perp, n_par, n_perp, n_par)
    
    v1_par_4d = vPar_1d[np.newaxis, :, np.newaxis, np.newaxis]  # shape (1, n_par, 1, 1)
    v1_perp_4d = vPerp_1d[:, np.newaxis, np.newaxis, np.newaxis]  # shape (n_perp, 1, 1, 1)
    v2_par_4d = vPar_1d[np.newaxis, np.newaxis, np.newaxis, :]  # shape (1, 1, 1, n_par)
    v2_perp_4d = vPerp_1d[np.newaxis, np.newaxis, :, np.newaxis]  # shape (1, 1, n_perp, 1)
    
    # Distribution functions in 4D
    f1_4d = f[:, :, np.newaxis, np.newaxis]  # shape (n_perp, n_par, 1, 1)
    f2_4d = f[np.newaxis, np.newaxis, :, :]  # shape (1, 1, n_perp, n_par)
    
    # print("Computing relative velocities...")
    
    # Calculate relative velocities
    # Relative parallel velocity
    v_rel_par = v1_par_4d - v2_par_4d  # broadcasts to (n_perp, n_par, n_perp, n_par)
    
    # Relative perpendicular velocity (averaged over azimuthal angle)
    v_rel_perp_sq_avg = v1_perp_4d**2 + v2_perp_4d**2
    
    # Relative speed
    v_rel = np.sqrt(v_rel_par**2 + v_rel_perp_sq_avg)
    
    # print("Computing center-of-mass energies...")
    
    # Center-of-mass energy in eV
    E_cm = 0.5 * m_reduced * v_rel**2 / (sc.constants.e)
    
    # Cross-section in m^2
    # Flatten for interpolation, then reshape back
    E_cm_flat = E_cm.flatten()
    sigma_flat = ddptFusionCXFunc(E_cm_flat)
    sigma = sigma_flat.reshape(E_cm.shape)
    
    # print("Computing integrand...")
    
    # Integrand: f1 * f2 * v_rel * sigma * (Jacobian factors)
    # Jacobian: 2*pi*v1_perp * 2*pi*v2_perp for both particles
    integrand = (f1_4d * f2_4d * v_rel * sigma * 
                 (2 * np.pi * v1_perp_4d) * (2 * np.pi * v2_perp_4d))
    
    # print("Integrating...")
    
    # Perform 4D integration using trapezoidal rule
    # Integrate over each dimension sequentially
    dv_par = vPar_1d[1] - vPar_1d[0]
    dv_perp = vPerp_1d[1] - vPerp_1d[0]
    
    # Integrate over v2_par (axis 3)
    result = np.trapezoid(integrand, dx=dv_par, axis=3)
    # Integrate over v2_perp (axis 2, which is now axis 2 after prev reduction)
    result = np.trapezoid(result, dx=dv_perp, axis=2)
    # Integrate over v1_par (axis 1)
    result = np.trapezoid(result, dx=dv_par, axis=1)
    # Integrate over v1_perp (axis 0)
    reactivity = np.trapezoid(result, dx=dv_perp, axis=0)
    
    # For same species (D-D), divide by 2 to avoid double counting
    reactivity /= 2
    
    # print("Integration complete!")
    
    return reactivity

def compute_row_reactivity(i, f_rz, vPar_1d, vPerp_1d):
    """
    Calculates the fusion reactivities for each value in the row. This is 
    purely a helper function that allows fusion_reactivity_rz to run
    things in parallel.
    """
    row = np.zeros(f_rz.shape[1])
    for j in range(f_rz.shape[1]):
        row[j] = fusion_reactivity(f_rz[i, j], vPar_1d, vPerp_1d)
    return i, row

def fusion_reactivity_rz(vPar, vPerp, zArr2D, rArr2D, f_rz, makeplot=False, savename=''):
    """
    Wrapper function to calculate the fusion reactivity / unit volume for a
    given set of distribution functions on r-z.

    Parameters
    ----------
    vPar : 2D numpy array  
        Parallel velocity grid in m/s, shape (len(vPar), len(vPerp))
    vPerp : 2D numpy array
        Perpendicular velocity grid in m/s, shape (len(vPar), len(vPerp))
    zArr2D : np.array
        z-locations where the f has been calculated. [meters]
    rArr2D : np.array
        r-locations where the f has been calculated. [meters].
    f_rz : np.array
        Distribution function at all points in the plasma.
        This is a 4D array with the following index convention-
        1st index - r-position
        2nd index - z-position
        3rd index - v_||
        4th index - v_perp
    makeplot : boolean, optional
        Make a plot of the fusion cross section. The default is False.
    savename : boolean, optional
        Save the reactivity data. The default is ''.

    Returns
    -------
    zArr2D : np.array
        z-locations where the reactivity has been calculated. [meters]
    rArr2D : np.array
        r-locations where the reactivity has been calculated. [meters].
    reactivity2D : np.array
        Fusion reactivity on the r-z grid. [counts / (m^3 s)]
    """
    
    # Create the dd fusion cross section interpolation function
    fusion_cross_section()

    # Extract the 1D velocity arrays from the 2D arrays
    vPar_1d = vPar[0, :]
    vPerp_1d = vPerp[:, 0]

    reactivity2D = np.zeros_like(zArr2D)
    
    # Setup so that compute_row can only be called with i. The rest of the arguments are pre-setup.
    worker = partial(compute_row_reactivity, f_rz=f_rz, vPar_1d=vPar_1d, vPerp_1d=vPerp_1d)

    # Use half the lana cores to parallelize the workflow
    with Pool(int(cpu_count()/2)) as pool:
            
        results = pool.map(worker, range(zArr2D.shape[0]))

        for i, row in results:
            reactivity2D[i] = row

    if makeplot == True:
        
        fig = plt.figure(figsize=(10, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        pltObj = ax.contourf(zArr2D, rArr2D, reactivity2D, levels=100, cmap='inferno')
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R[m]')
        
        cbar = fig.colorbar(pltObj, label=r'Reactivity [counts/(m$^3$s)]')
        
        plt.show()
        
    if savename != '':
        
        np.savez(savename, 
                 rArr2D = rArr2D,
                 zArr2D = zArr2D,
                 reactivity2D = reactivity2D)
    
    return zArr2D, rArr2D, reactivity2D

# Load pleiades output
if __name__ == '__tempmain__':
    """
    Load the distribution function from a pleaides output.
    """

    # Pleaides output file location
    filenamePleiades = '/home/sanwalka/synthetic_proton_detector/data/pleiades_260105053.h5'

    # eqdsk output file location
    filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    vPar, vPerp, zArr2D, rArr2D, f_rz = dist_func_rz_pleiades(filenamePleiades, filenameEqdsk)

# Plot f at midplane
if __name__ == '__tempmain__':
    """
    Plot the distribution function at the midplane
    """

    #########################################################################################
    #### HTPD 2026 Plot
    # Save a plot of the distribution function at z=0 for a 0.1keV Te case
    _, _, _ = dist_func(n_fast = 5e18,
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
                        savename = 'dist_func_with_maxwellian.png')
    #########################################################################################

    # Location of the CQL3D output file
    # filenameCQL3D = '/home/sanwalka/synthetic_proton_detector/WHAM_R_2gen_6e18_50eV_w_add_source.nc'

    # vPar, vPerp, zArr2D, rArr2D, f_rz = dist_func_rz_cql3d(filenameCQL3D)

# Plot f at z-positions
if __name__ == '__tempmain__':
    """
    Plot the distribution function at 2 z-positions
    """

    #########################################################################################
    #### HTPD 2026 Plot
    vPar, vPerp, f = dist_func(n_fast = 5e18,
                               n_max = 5e19,
                               T_max = 1,
                               R_m = 57,
                               E_NBI = 25,
                               theta_NBI = np.pi/4,
                               T_e = 0.075,
                               mu_i = 2,
                               Z_eff = 3,
                               gridsize = 500)

    print('Maximum values of the midlane distribution function-')
    print(vPar.max(), vPerp.max(), f.max())
    print('****************************************************')

    print('Minimum values of the midlane distribution function-')
    print(vPar.min(), vPerp.min(), f.min())
    print('****************************************************')
    
    # Location of magnetic equilibrium file
    filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    # Load the magnetic equilibrium
    Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk)

    # Radial distance where the distribution function is calculated from the midplane
    rDist = 0.01 # [m]

    # z-positions where we want the evolved distribution function
    zValArr = np.linspace(0, 0.8, 20)

    # Calculate the z-evolved distribution function
    _, _ = dist_func_z_evol(f, vPar, vPerp, rDist, zValArr, Rmesh, Zmesh, Bmag, magneticFlux, makeplot=True)

# Calculate reactivity along z for 1 flux tube
if __name__ == '__tempmain__':
    """
    Test to see if the fusion reactivity calculator is working properly
    """

    # =============================================
    # Test with a maxwellian
    # =============================================

    # Generate a maxwellian with 10keV
    vPar, vPerp, f = dist_func(n_fast = 5,
                               n_max = 5e19,
                               T_max = 10,
                               R_m = 57,
                               E_NBI = 25,
                               theta_NBI = np.pi/4,
                               T_e = 0.075,
                               mu_i = 2,
                               Z_eff = 3,
                               gridsize = 50,
                               makeplot = True)
    
    vPar_1d = vPar[0, :]
    vPerp_1d = vPerp[:, 0]

    fusion_cross_section()
    reactivity = fusion_reactivity(f, vPar_1d, vPerp_1d)

    print(reactivity)
    
    # =============================================
    # Test with a z-evolved maxwellian
    # =============================================

    # Generate a maxwellian at the midplane with 10keV
    vPar, vPerp, f = dist_func(n_fast = 5,
                               n_max = 5e19,
                               T_max = 10,
                               R_m = 57,
                               E_NBI = 25,
                               theta_NBI = np.pi/4,
                               T_e = 0.075,
                               mu_i = 2,
                               Z_eff = 3,
                               gridsize = 50)

    vPar_1d = vPar[0, :]
    vPerp_1d = vPerp[:, 0]

    # Location of magnetic equilibrium file
    filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    # Load the magnetic equilibrium
    Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk)

    # z-positions where we want to calculate the reactivity
    zArr = np.linspace(0, 0.8, 50)

    # Evolve f along z
    zEvolF, rValArr = dist_func_z_evol(f, vPar, vPerp, 0, zArr, Rmesh, Zmesh, Bmag, magneticFlux)

    reactivity1D = np.zeros(shape=len(zArr))

    # Calculate the reactivity at each z position
    for i in range(len(zArr)):

        reactivity1D[i] = fusion_reactivity(zEvolF[i], vPar_1d, vPerp_1d)

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
    vPar, vPerp, f = dist_func(n_fast = 5e18,
                               n_max = 5e19,
                               T_max = 1,
                               R_m = 57,
                               E_NBI = 25,
                               theta_NBI = np.pi/4,
                               T_e = 0.075,
                               mu_i = 2,
                               Z_eff = 3,
                               gridsize = 50)

    vPar_1d = vPar[0, :]
    vPerp_1d = vPerp[:, 0]

    # Evolve f along z
    zEvolF, rValArr = dist_func_z_evol(f, vPar, vPerp, 0, zArr, Rmesh, Zmesh, Bmag, magneticFlux)

    reactivity1D = np.zeros(shape=len(zArr))

    # Calculate the reactivity at each z position
    for i in range(len(zArr)):

        reactivity1D[i] = fusion_reactivity(zEvolF[i], vPar_1d, vPerp_1d)

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
    
# Calculate 2D reactivity profile
if __name__ == '__tempmain__':
    """
    Calculate the 2D fusion reactivity profile
    """
    
    # Location of magnetic equilibrium file
    filenameEqdsk = '/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    
    # Location to store the 2D fusion reactivity profile
    savenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_faster_interpolator.npz'
    
    # Get the r-z evolved distribution functions
    rArr = np.linspace(0, 0.1, 5)
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
    vPar, vPerp, zArr2D, rArr2D, f_rz = dist_func_rz(rArr, 
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
    
    # print('finished calculating the distribution functions')
    
    # Get the r-z evolved fusion reactivity profile
    zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(vPar, vPerp, zArr2D, rArr2D, f_rz, 
                                                        makeplot=True,
                                                        savename=savenameReactivity)
    reactivityTime = time.time()
    print(f'Time taken to generate the reactivity = {np.round(reactivityTime - distFuncTime, 2)}s')
    
    # Get the fusion cross section
    # fusion_cross_section(makeplot=True)
    
    # Get the distribution function data
    # vPar, vPerp, f = dist_func(1e19, 57, 25, np.pi/4, 0.1, 2, 3, gridsize=100, makeplot=True)
    
    # Calculate the fusion reactivity of the distribution function
    # reactivity = fusion_reactivity(f, vPar, vPerp)
    
    # Get the magnetic equilibrium data
    # Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk,
    #                                                                 makeplot=True)
    
    # Get the z-evolved distribution functions
    zValArr = np.linspace(0, 0.8, 20)
    rDist = 0.1
    # zEvolf = dist_func_z_evol(f, vPar, vPerp, rDist, zValArr, Rmesh, Zmesh, Bmag, magneticFlux, makeplot=True)
