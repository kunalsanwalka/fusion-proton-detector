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
    Y : np.array
        Velocity array.
        To get the 1D array, use Y[0,:].
        Units - m/s
    X : np.array
        Normalized pitch angle array.
        To get the 1D array, use X[:,0].
        Units - normalized. 0 = v_perp ; 1 = v_par
    f : np.array
        Density normalized distribution function.
        1st index - pitch angle
        2nd index - velocity
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
        
    # Define the velocity and pitch angle grids
    v = np.linspace(0, vNBI, gridsize) # velocity [m/s]
    xsi = np.linspace(0, 1, gridsize) # pitch angle [normalized] 0 = v_perp ; 1 = v_par
    Y, X = np.meshgrid(v, xsi)

    # Obtain the distribution function
    fFast = fObj.f(Y, X.copy())

    # Normalize the fast ion distribution function to n_fast
    integrand = 2*np.pi * fFast * (Y**2) # spherical coordinates jacobian
    fastDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), v, axis=0) * 2
    fFast *= n_fast/fastDens

    if makeplot:
        # Check the integral
        integrand = 2*np.pi * fFast * (Y**2)
        fastDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), v, axis=0) * 2
        print(f'0th moment of the fast ion distribution function = {fastDens}'+r'm$^{-3}$')
        print(f'Input fast ion density = {n_fast}'+r'm$^{-3}$')

    # ======================================
    # Maxwellian background component
    # ======================================

    tempInJoules = T_max*1e3*q
    ionMass = m_p*mu_i

    fMaxwell = np.exp(-ionMass*Y**2/(2*tempInJoules))

    # Normalize the Maxwellian distribution function to n_max
    integrand = 2 * np.pi * fMaxwell * (Y**2) # spherical coordinates jacobian
    maxwellDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), v, axis=0) * 2
    fMaxwell *= n_max/maxwellDens

    if makeplot:
        # Check the integral
        integrand = 2*np.pi * fMaxwell * (Y**2)
        maxwellDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), v, axis=0) * 2
        print(f'0th moment of the maxwellian ion distribution function = {maxwellDens}'+r'm$^{-3}$')
        print(f'Input maxwellian ion density = {n_max}'+r'm$^{-3}$')

    # ======================================    
    # Combined distribution function
    # ======================================

    f = fFast + fMaxwell
    f = np.clip(f, a_min=0, a_max=None)

    if makeplot == True:
        
        # Convert the velocity grids to vPar and vPerp for plotting
        vPar = Y * X
        vPerp = Y * np.sqrt(1-X**2)

        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Normalize first, then clip to log-safe minimum BEFORE passing to contourf
        f_norm = f / np.max(f)
        vmin_log = 1e-6
        f_norm = np.clip(f_norm, a_min=vmin_log, a_max=1)  # kill exact zeros
        
        levels = np.logspace(np.log10(vmin_log), 0, 100)   # log-spaced levels
        
        pltObj = ax.contourf(vPar/vNBI, vPerp/vNBI, f_norm,
                             norm=colors.LogNorm(vmin=vmin_log, vmax=1),
                             levels=levels,
                             extend='neither')
        
        cbar = fig.colorbar(pltObj, ax=ax)
        cbar.set_label(r'f/f$_{max}$')
        cbar.set_ticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        cbar.set_ticklabels([r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        
        # Add a text box with the plasma parameters
        textstr = '\n'.join((
            r'$E_{NBI}$ = '+str(E_NBI)+r' keV',
            r'$T_e$ = '+str(int(T_e*1e3))+r' eV',
            r'$Z_{eff}$ = '+str(Z_eff),
            r'$R_m$ = '+str(R_m)))
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=22,
                horizontalalignment='right', verticalalignment='top', bbox=props)
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        ax.set_title(r'$f_{z=0}(v_{||}, v_{\perp})$')
        
        if savename is not None:
            plt.savefig(plotDir+savename, dpi=300)

        plt.show()
        
    return Y, X, f

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
    # Resolved relative to this file (not the caller's cwd) so it still works
    # when called from driver scripts living outside the root directory.
    eqdskScriptPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eqdsk_analysis_functions.py')
    subprocess.run(['/share/envs/pleiades_env/bin/python',
                    eqdskScriptPath,
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

def _trace_flux_tube(rDist, zValArr, Rmesh, Zmesh, Bmag, magneticFlux, BmagInterpolator):
    """
    Follow a single flux tube from its midplane starting radius rDist out to
    every z in zValArr, and report how r and B/B0 change along the way.

    A flux tube is a curve of constant magnetic flux, so "tracing" it means
    finding the flux contour that passes through (z=0, r=rDist) and reading
    off that contour's r-value at each requested z. This r(z), B(z)/B0
    mapping is exactly the geometry information dist_func_z_evol needs to
    apply mu-conservation, and it used to be recomputed from scratch every
    time a distribution function was evolved. It's now factored out here so
    build_flux_tube_geometry can compute it once per flux surface.

    Parameters
    ----------
    rDist : float
        Midplane (z=0) radial starting position of the flux tube. [m]
    zValArr : np.array
        z-positions to evaluate the flux tube at. [m]
    Rmesh, Zmesh : np.array
        2D meshes of radial/axial positions from magnetic_equilibrium.
    Bmag : np.array
        |B| on the (Zmesh, Rmesh) grid. [Tesla]
    magneticFlux : np.array
        Magnetic flux on the (Zmesh, Rmesh) grid. [Weber]
    BmagInterpolator : scipy.interpolate.RegularGridInterpolator
        Interpolator for Bmag over (z, r). Built once by
        build_flux_tube_geometry and passed in so it isn't rebuilt for
        every flux tube.

    Returns
    -------
    rValArr : np.array
        r-position of the flux tube at each z in zValArr. [m]
    bNormArr : np.array
        B(z)/B(z=0) along the flux tube.
    Rm : float
        Mirror ratio of this flux tube: field strength at the mirror
        throat divided by the midplane field strength.
    """

    # Look up the flux value at (z=0, rDist), then contour the whole 2D
    # flux map at that level to get the flux surface passing through it
    idx = np.abs(Rmesh[0] - rDist).argmin()
    midplaneFlux = magneticFlux[int(len(Zmesh)/2), idx]

    # The flux surface of interest is a single closed/connected curve, so
    # just take the first contour skimage finds at this level
    contours = skimage.measure.find_contours(magneticFlux, level=midplaneFlux)
    contour = contours[0]

    # find_contours returns fractional indices — interpolate into the real
    # coordinate arrays rather than rounding to the nearest grid point
    z_axis = Zmesh[:, 0]
    r_axis = Rmesh[0]

    z_idx_interp = sc.interpolate.interp1d(np.arange(len(z_axis)), z_axis)
    r_idx_interp = sc.interpolate.interp1d(np.arange(len(r_axis)), r_axis)
    z_vals = z_idx_interp(contour[:, 0])
    r_vals = r_idx_interp(contour[:, 1])

    # r as a function of z along this flux tube, so it can be sampled at
    # both the requested zValArr and later at the mirror throat
    contour_interp = sc.interpolate.interp1d(z_vals, r_vals, bounds_error=False, fill_value=np.nan)
    rValArr = contour_interp(zValArr)

    # Field strength along the traced path, normalized to the midplane
    # value to give B(z)/B0
    points = np.array([zValArr, rValArr]).T
    bMagArr = BmagInterpolator(points)
    bNorm = BmagInterpolator([0, rDist])[0]
    bNormArr = bMagArr / bNorm

    # Mirror ratio: field strength at the throat (95% of the z-domain,
    # since the equilibrium data can be noisy right at the grid edge)
    # relative to the midplane
    zThroat = np.max(Zmesh) * 0.95
    rAtThroat = contour_interp(zThroat)
    bMagInThroat = BmagInterpolator([zThroat, rAtThroat])[0]
    Rm = bMagInThroat / bNorm

    return rValArr, bNormArr, Rm

def build_flux_tube_geometry(rArr, zValArr, Rmesh, Zmesh, Bmag, magneticFlux):
    """
    One-time precomputation: traces the flux tube and B(z)/B0 profile for
    every starting radius in rArr. None of this depends on the particle
    distribution (density, temperature, etc.), only on rArr and the
    magnetic equilibrium, so it's computed once before the parallel sweep
    in dist_func_rz starts instead of being redone inside every worker.

    Parameters
    ----------
    rArr : np.array
        Midplane radial positions to trace flux tubes from. [m]
    zValArr : np.array
        z-positions to evaluate each flux tube at. [m]
    Rmesh, Zmesh : np.array
        2D meshes of radial/axial positions from magnetic_equilibrium.
    Bmag : np.array
        |B| on the (Zmesh, Rmesh) grid. [Tesla]
    magneticFlux : np.array
        Magnetic flux on the (Zmesh, Rmesh) grid. [Weber]

    Returns
    -------
    rAlongTube_all : np.array
        Shape (len(rArr), len(zValArr)). r-position of each flux tube at
        each z. [m]
    bNormArr_all : np.array
        Shape (len(rArr), len(zValArr)). B(z)/B0 along each flux tube, fed
        into dist_func_z_evol to apply mu-conservation.
    RmArr : np.array
        Shape (len(rArr),). Mirror ratio of each flux tube. Passed into
        dist_func as R_m so the midplane distribution function for each
        flux surface is generated with a mirror ratio consistent with its
        own field geometry, rather than a single value shared by all r.
    """

    # Geometry-only, so build once here and reuse for every flux tube
    # instead of rebuilding it inside _trace_flux_tube on every call
    BmagInterpolator = sc.interpolate.RegularGridInterpolator((Zmesh[:, 0], Rmesh[0]), Bmag)

    n_r, n_z = len(rArr), len(zValArr)
    rAlongTube_all = np.empty((n_r, n_z))
    bNormArr_all   = np.empty((n_r, n_z))
    RmArr          = np.empty(n_r)

    for i, rDist in enumerate(rArr):

        # Radial position along each flux tube
        rAlongTube_all[i], bNormArr_all[i], RmArr[i] = _trace_flux_tube(rDist, zValArr, Rmesh, Zmesh, Bmag, magneticFlux, BmagInterpolator)

    return rAlongTube_all, bNormArr_all, RmArr

def dist_func_z_evol(f, vel, xsi, bNormArr, zValArr=None, makeplot=False):
    """
    This function evolves the distribution along z for a given magnetic
    equilibrium. It does this via the mu (1st adiabatic invariant)
    conservation.

    Parameters
    ----------
    f : np.array
        2D distribution function.
    vel : np.array
        Velocity array.
        To get the 1D array, use Y[0,:].
        Units - m/s
    xsi : np.array
        Normalized pitch angle array.
        To get the 1D array, use X[:,0].
        Units - normalized. 0 = v_perp ; 1 = v_par
    bNormArr: np.array
        Normalized magnetic field strength along the flux tube, i.e.
        B(z)/B(z=0). Precomputed once per flux tube by
        build_flux_tube_geometry / _trace_flux_tube rather than here, so
        this function is now pure mu-conservation math with no equilibrium
        interpolation of its own.
    zValArr : np.array, optional
        z-positions corresponding to bNormArr, used only to label the
        makeplot titles. If None, the plot titles report B/B0 instead of z.
        The default is None.
    makeplot : boolean, optional
        Make a plot of the distribution function at 2 z-positions to make sure
        the function is working properly. The default is False.

    Returns
    -------
    zEvolf : np.array
        3D array with the z-evolved distribution function, one 2D f per
        z-position in bNormArr (index 0 = z-position).
    """

    # 3D array to store the z-evolved distribution function
    zEvolf = np.zeros(shape=(len(bNormArr), *f.shape))
    xsi_1d = xsi[:, 0]
    
    for i in range(len(bNormArr)):
    
        b = bNormArr[i]

        # xsi(z)^2 = 1 - b*(1 - xsi0^2); negative => particle mirrors before
        # reaching this z (trapped), so exclude it.
        arg = 1.0 - b * (1.0 - xsi_1d**2)
        valid = arg >= 0
 
        xsiNew = np.sign(xsi_1d) * np.sqrt(np.clip(arg, 0, None))
 
        xsiNew_valid = xsiNew[valid]
        f_valid = f[valid, :]
 
        # Sort by the new xsi value (robust regardless of xsi_1d's original
        # ordering, and correctly stitches together the +/- branches)
        order = np.argsort(xsiNew_valid)
        xsiNew_sorted = xsiNew_valid[order]
        f_sorted = f_valid[order, :]
 
        # One interpolation call handles every v column at once, since the
        # xsi remap doesn't depend on v
        interp = sc.interpolate.interp1d(xsiNew_sorted, f_sorted, 
                                         axis=0, 
                                         kind='linear', 
                                         bounds_error=False, 
                                         fill_value=(f_sorted[0], f_sorted[-1]))
 
        result = interp(xsi_1d)

        # Clean up and add to zEvolf
        result[result < 0] = 0.0
        zEvolf[i] = result
    
    if makeplot == True:
        
        # Convert the velocity grids to vPar and vPerp for plotting
        vPar = vel * xsi
        vPerp = vel * np.sqrt(1-xsi**2)

        fig = plt.figure(figsize=(12, 16), tight_layout=True)

        #### Midplane plot
                
        ax = fig.add_subplot(211)
        
        # Normalize first, then clip to log-safe minimum BEFORE passing to contourf
        index = 0
        normVal = np.max(zEvolf[index])
        f_norm = zEvolf[index] / normVal
        vmin_log = 1e-6
        f_norm = np.clip(f_norm, a_min=vmin_log, a_max=1)  # kill exact zeros
        
        levels = np.logspace(np.log10(vmin_log), 0, 100)   # log-spaced levels

        pltObj = ax.contourf(vPar/np.max(vPar), vPerp/np.max(vPerp), f_norm, 
                             norm=colors.LogNorm(vmin=vmin_log, vmax=1),
                             levels=levels,
                             extend='neither')
        
        cbar = fig.colorbar(pltObj, ax=ax)
        cbar.set_label(r'f/f$_{max}$')
        cbar.set_ticks([1e-6, 1e-4, 1e-2, 1e0])
        cbar.set_ticklabels([r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$'])
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        title = f'z = {np.round(zValArr[index], 2)}m' if zValArr is not None else f'B/B0 = {np.round(bNormArr[index], 2)}'
        ax.set_title(title, pad=45)

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
        cbar.set_ticks([1e-6, 1e-4, 1e-2, 1e0])
        cbar.set_ticklabels([r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$'])
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$v_{||}/v_0$')
        ax.set_ylabel(r'$v_{\perp}/v_0$')
        title = f'z = {np.round(zValArr[index], 2)}m' if zValArr is not None else f'B/B0 = {np.round(bNormArr[index], 2)}'
        ax.set_title(title, pad=45)
        
        plt.show()
    
    return zEvolf

def compute_row_fz(i):
    """
    Computes f along z for a given set of r and z values. This is purely a
    helper function that allows dist_func_rz to run things in
    parallel.

    Takes only the row index i and reads everything else (the profile
    arrays, NBI parameters, and the precomputed flux tube geometry from
    build_flux_tube_geometry) from module globals set by _init_worker_fz.
    This means pool.map only has to pickle an int per task, while the
    bulkier shared arrays are sent to each worker process once via the
    Pool initializer instead of once per row of rArr.
    """

    # Get the distribution function at the midplane
    vel, xsi, midplaneF = dist_func(_nFastArr[i], _nMaxArr[i], _TMaxArr[i], _RmArr[i], _E_NBI, _theta_NBI, _TeArr[i], _mu_i, _ZeffArr[i], gridsize=_gridsize)
    
    # Evolve it along zVals
    f_z = dist_func_z_evol(midplaneF, vel, xsi, _bNormArr_all[i])
    
    return i, f_z, _rAlongTube_all[i], vel, xsi

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
    vel : np.array
        velocity array. [m/s]
    xsi : np.array
        normalized pitch angle array. [m/s]
    zArr2D : np.array
        z-locations where the f has been calculated. [meters]
    rArr2D : np.array
        r-locations where the f has been calculated. [meters].
    f_rz : np.array
        Distribution function at all points in the plasma.
        This is a 4D array with the following index convention-
        1st index - r-position
        2nd index - z-position
        3rd index - v
        4th index - xsi
    """
    
    # Global variables that help remove redundant computation in the parallelization workflow
    _bNormArr_all = None
    _rAlongTube_all = None
    _nFastArr = None
    _nMaxArr = None
    _TMaxArr = None
    _RmArr = None
    _E_NBI = None
    _theta_NBI = None
    _TeArr = None
    _mu_i = None
    _ZeffArr = None
    _gridsize = None

    # Pool initializer: runs once per worker process (not once per task) and
    # stashes the shared arrays as module globals, so compute_row_fz can be
    # called with just an index i instead of having every one of the
    # len(rArr) tasks re-pickle and re-send the full arrays.
    def _init_worker_fz(bNormArr_all, rAlongTube_all, nFastArr, nMaxArr, TMaxArr, RmArr, E_NBI, theta_NBI, TeArr, mu_i, ZeffArr, gridsize):

        global _bNormArr_all, _rAlongTube_all, _nFastArr, _nMaxArr, _TMaxArr, _RmArr, _E_NBI, _theta_NBI, _TeArr, _mu_i, _ZeffArr, _gridsize

        _bNormArr_all = bNormArr_all 
        _rAlongTube_all = rAlongTube_all
        _nFastArr = nFastArr
        _nMaxArr = nMaxArr
        _TMaxArr = TMaxArr 
        _RmArr = RmArr
        _E_NBI = E_NBI 
        _theta_NBI = theta_NBI 
        _TeArr = TeArr
        _mu_i = mu_i 
        _ZeffArr = ZeffArr 
        _gridsize = gridsize

    # =========================================================================
    # Calculate the mirror ratios for each flux surface as defined by rArr for
    # the input magnetic equilibrium.
    # =========================================================================
    
    # Load the magnetic equilibirum
    Rmesh, Zmesh, Br, Bz, Bmag, magneticFlux = magnetic_equilibrium(filenameEqdsk,
                                                                    makeplot=False)

    # Trace all flux tubes once here, before the parallel sweep, instead of
    # inside each worker (see build_flux_tube_geometry for why)
    rAlongTube_all, bNormArr_all, RmArr = build_flux_tube_geometry(rArr, zArr, Rmesh, Zmesh, Bmag, magneticFlux)
       
    # =========================================================================
    # Generate the midplane distribution functions and then evolve them along
    # zArr to get a 2D map of the distribution functions
    # =========================================================================
    
    # Size of the velocity grid
    gridsize = 50

    # This will be a whopper array that will store all the distribution functions
    f_rz = np.zeros(shape=(len(rArr), len(zArr), gridsize, gridsize))
    vel = None
    xsi = None
    rArr2D = []
    
    # Setup so that compute_row can only be called with i. The rest of the arguments are pre-setup.

    # Change the plotting backend to be non-interactive for the worker function since it will be called in parallel and we don't want multiple plot windows popping up.
    plt.switch_backend('Agg')

    # Use half of the cores on lana
    with Pool(int(cpu_count()/2), initializer=_init_worker_fz, initargs=(bNormArr_all, rAlongTube_all, nFastArr, nMaxArr, TMaxArr, RmArr, E_NBI, theta_NBI, TeArr, mu_i, ZeffArr, gridsize)) as pool:
        results = pool.map(compute_row_fz, range(len(rArr)))

        # Unpack results
        for i, f_z, rAlongTube, vel, xsi in results:
            f_rz[i] = f_z
            rArr2D.append(rAlongTube)
            vel = vel
            xsi = xsi

    # Change the plotting backend back to interactive for the rest of the code.
    plt.switch_backend('TkAgg')

    rArr2D = np.array(rArr2D)
    
    # Make the z-array also 2D to make the analysis easier
    zArr2D = np.tile(zArr, (len(rArr), 1))
    
    return vel, xsi, zArr2D, rArr2D, f_rz

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

    # ddptFusionCXFunc = sc.interpolate.CubicSpline(energyGrid, DDa_xs)

    # The cross section spans many orders of magnitude and rises roughly as
    # a power law with energy near threshold, so it is much closer to a
    # straight line in log(E)-log(sigma) space than in linear space.
    # Interpolating there is both more accurate than linear interpolation
    # and cheaper to evaluate than re-fitting/evaluating a CubicSpline
    # object, which matters because build_fusion_kernel calls this once per
    # (v1,xsi1,v2,xsi2) grid point pair.
    log_E = np.log(energyGrid)
    log_sigma = np.log(np.maximum(DDa_xs, 1e-300))  # floor avoids log(0) at threshold

    def ddptFusionCXFunc(E):
        # Clamp instead of extrapolating past the tabulated cross section data
        E_clamped = np.clip(E, energyGrid[0], energyGrid[-1])
        return np.exp(np.interp(np.log(E_clamped), log_E, log_sigma))
    
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

def extend_f(f, v, xsi, makeplot=False):
    """
    This function extends the domain of xsi from [0, 1] to [-1, 1].

    This is important for fusion_reactivity where we need to account for the co and counter propagating particles.
    The fusion cross section is very sensitive to the relative velocities of the particles.

    Parameters
    ----------
    f : np.array
        Density normalized distribution function.
        1st index - pitch angle (xsi)
        2nd index - velocity (v)
    v : 1D numpy array  
        Velocity grid in m/s
    xsi : 1D numpy array
        Normalized pitch angle.
        Domain must be from [0, 1]

    Returns
    -------
    f : np.array
        Density normalized distribution function.
        1st index - pitch angle (xsi)
        2nd index - velocity (v)
    v : 1D numpy array  
        Velocity grid in m/s
    xsi : 1D numpy array
        Normalized pitch angle. Now extended to [-1, 1]
    """

    if makeplot:
        # Check normalization before extension

        Y, _ = np.meshgrid(v, xsi)
        integrand = 2*np.pi * f * (Y**2)
        maxwellDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), v, axis=0)

        print(f'density before extension = {maxwellDens}'+r'm$^{-3}$')
        
    #### xsi input domain is [0,1], extend the domain to [-1, 1] to account for counter-propagating particles
    
    # Negative branch excludes xi=0 so that column isn't duplicated
    xi_neg = -xsi[1:][::-1]
    xi_full_1d = np.concatenate([xi_neg, xsi])
    
    f_neg = f[1:, :][::-1, :]
    f_full = np.concatenate([f_neg, f], axis=0) / 2 # keep the total density the same

    if makeplot:
        # Check normalization after extension

        Y, _ = np.meshgrid(v, xi_full_1d)
        integrand = 2*np.pi * f_full * (Y**2)
        maxwellDens = np.trapezoid(np.trapezoid(integrand, xi_full_1d, axis=0), v, axis=0)

        print(f'density after extension = {maxwellDens}'+r'm$^{-3}$')

    return f_full, v, xi_full_1d

def _trapz_weights(x):
    """
    Quadrature weight vector for the composite trapezoidal rule on a
    uniformly spaced grid x, such that sum(w * y) == np.trapezoid(y, x).

    Every interior point covers a full grid spacing dx; the two endpoints
    are the outer edge of only one trapezoid instead of two, so they get
    half weight. This lets build_fusion_kernel bake the entire trapezoidal
    integration scheme into the kernel matrix once, instead of calling
    np.trapezoid on every evaluation of fusion_reactivity.

    Parameters
    ----------
    x : 1D np.array
        Uniformly spaced grid.

    Returns
    -------
    w : 1D np.array
        Trapezoidal weight for each point in x.
    """

    dx = x[1] - x[0]

    w = np.full(len(x), dx)

    # Endpoints border only one trapezoid instead of two
    w[0] *= 0.5
    w[-1] *= 0.5

    return w

def build_fusion_kernel(v, xsi, chunk_size=4):
    """
    One-time precomputation of the fusion reactivity kernel matrix K.

    fusion_reactivity computes a double integral over velocity space of
    f(v1,xsi1) * f(v2,xsi2) * sigma(v_rel) * v_rel, discretized as four
    nested trapezoidal sums (see the pre-refactor implementation: it looped
    over v1, xsi1, v2, xsi2 with np.trapezoid). Nothing in that integrand
    except f itself depends on the plasma state — the quadrature weights,
    the spherical-coordinate jacobians, the relative velocity between grid
    points, and the cross section sigma(v_rel) are all fixed once the
    (v, xsi) grid is fixed. So the whole integral is bilinear in f (f
    appears exactly twice) and can be written as a single quadratic form
    f^T K f, where K packs in everything that doesn't depend on f. Since
    dist_func_rz reuses the same (v, xsi) grid for every (r, z) point, K
    only needs to be built once for the whole sweep — fusion_reactivity
    then reduces to one matrix-vector product per distribution function
    instead of a full 4D integral.

    v, xsi must be the exact grids you will use for every subsequent call
    to fusion_reactivity (K is only valid for the grid it was built from).

    Parameters
    ----------
    v : 1D np.array
        Velocity grid in m/s (pre-extension, i.e. what dist_func returns).
    xsi : 1D np.array
        Normalized pitch-angle grid over [0, 1] (pre-extension).
    chunk_size : int, optional
        Number of xsi1 (particle-1 pitch-angle) values to process per
        iteration. Building all pairwise combinations at once would need a
        4D temporary array of shape (n_xi, n_v, n_xi, n_v), which is too
        large to hold in memory for a reasonably fine grid; chunking over
        xsi1 bounds the peak memory use at the cost of a python-level loop.

    Returns
    -------
    K : np.array
        Symmetric (N x N) kernel matrix, N = len(xsi_ext) * len(v_ext).
        Row/column order matches f_ext.ravel() from extend_f (xsi varies
        slowest, v fastest), so fusion_reactivity can flatten f the same
        way and compute f^T K f directly.
    v_ext : np.array
        Velocity grid extended to match xsi_ext (same length as v; kept
        for bookkeeping/consistency checks).
    xsi_ext : np.array
        Pitch-angle grid extended from [0, 1] to [-1, 1] by extend_f, to
        account for co- and counter-propagating particles.
    """

    # extend_f mirrors the grid geometry, not the values of f, so a
    # dummy f is fine here just to recover the extended grids.
    dummy_f = np.zeros((len(xsi), len(v)))
    _, v_ext, xsi_ext = extend_f(dummy_f, v, xsi)

    n_xi, n_v = len(xsi_ext), len(v_ext)
    N = n_xi * n_v
    # Reduced mass for identical (D-D) particles [kg], used to get the
    # center-of-mass collision energy from the relative velocity
    m_reduced = sc.constants.m_p / 2.0

    w_xi = _trapz_weights(xsi_ext)
    w_v  = _trapz_weights(v_ext) * 2 * np.pi * v_ext**2   # quadrature weight * spherical jacobian, fused

    # "Particle 2" grids, broadcast so they'll pair against every particle-1
    # grid point below (axes: xi1, v1, xi2, v2)
    xi2 = xsi_ext[np.newaxis, np.newaxis, :, np.newaxis]
    v2  = v_ext[np.newaxis, np.newaxis, np.newaxis, :]
    vpar2 = v2 * xi2
    vperp2_sq = v2**2 * (1 - xi2**2)
    w2 = (w_xi[np.newaxis, np.newaxis, :, np.newaxis] *
          w_v[np.newaxis, np.newaxis, np.newaxis, :])

    K = np.empty((N, N))

    # Fill K one block of xsi1 rows at a time to keep the (c, n_v, n_xi, n_v)
    # temporaries a manageable size
    for start in range(0, n_xi, chunk_size):
        stop = min(start + chunk_size, n_xi)
        c = stop - start

        # "Particle 1" grids for this chunk of xsi1 values
        xi1 = xsi_ext[start:stop, np.newaxis, np.newaxis, np.newaxis]
        v1  = v_ext[np.newaxis, :, np.newaxis, np.newaxis]
        vpar1 = v1 * xi1
        vperp1_sq = v1**2 * (1 - xi1**2)

        # Relative velocity between every particle-1/particle-2 grid point
        # pair in this chunk, and the corresponding cross section
        v_rel = np.sqrt((vpar1 - vpar2)**2 + vperp1_sq + vperp2_sq)
        E_cm = 0.5 * m_reduced * v_rel**2 / sc.constants.e
        sigma = ddptFusionCXFunc(E_cm.ravel()).reshape(E_cm.shape)

        w1 = (w_xi[start:stop, np.newaxis, np.newaxis, np.newaxis] *
              w_v[np.newaxis, :, np.newaxis, np.newaxis])

        # Collapse (xi1_chunk, v1, xi2, v2) -> (rows, columns). Row-major
        # reshape matches f_ext.ravel()'s (xsi, v) layout so that later
        # f_flat @ K @ f_flat lines up index-for-index with this block.
        chunk = (w1 * w2 * v_rel * sigma).reshape(c * n_v, N)
        K[start * n_v: stop * n_v, :] = chunk

    return K, v_ext, xsi_ext

def fusion_reactivity(f, v, xsi, K, symv):
    """
    Calculate fusion reactivity for a distribution function f, using the
    kernel K precomputed by build_fusion_kernel.

    Since the four nested trapezoidal integrals over velocity space reduce
    to the quadratic form f^T K f (see build_fusion_kernel for the
    derivation), this function no longer does any physics of its own — it
    just extends f the same way K's grid was extended, flattens it, and
    evaluates that quadratic form.

    Parameters
    ----------
    f : np.array
        Density normalized distribution function.
        1st index - pitch angle (xsi)
        2nd index - velocity (v)
    v : 1D numpy array
        Velocity grid in m/s. Must match the grid K was built from.
    xsi : 1D numpy array
        Normalized pitch angle over [0, 1]. Must match the grid K was
        built from.
    K : np.array
        Precomputed kernel matrix from build_fusion_kernel.
    symv : callable
        BLAS symmetric matrix-vector multiply function (from
        scipy.linalg.blas.get_blas_funcs), looked up once by the caller
        and passed in so it isn't re-resolved on every call.

    Returns
    -------
    reactivity : float
        Fusion reactivity Gamma in 1/(m^3*s).
    """

    f_ext, _, _ = extend_f(f, v, xsi)
    f_flat = np.ascontiguousarray(f_ext.ravel(), dtype=K.dtype)

    # K.T is mathematically identical to K (symmetric), but it's a
    # Fortran-contiguous *view* of a C-contiguous array, so this avoids
    # the copy that BLAS would otherwise make internally.
    Kf = symv(alpha=1.0, a=K.T, x=f_flat)

    # f^T K f, the discretized double integral over (v1,xsi1) and (v2,xsi2)
    return float(f_flat @ Kf)

def compute_row_reactivity(i):
    """
    Calculates the fusion reactivities for each value in the row. This is
    purely a helper function that allows fusion_reactivity_rz to run
    things in parallel.

    Reads K, the velocity/pitch-angle grids, and symv from module globals
    set by _init_worker rather than taking them as arguments, because
    pool.map only pickles the argument (i) for each of the zArr2D.shape[0]
    tasks — the large shared arrays are sent to each worker process once,
    via the Pool initializer, instead of being re-pickled for every row.
    """

    row = np.zeros(_f_rz.shape[1])

    for j in range(_f_rz.shape[1]):
        row[j] = fusion_reactivity(_f_rz[i, j], _vel_1d, _xsi_1d, _K, _symv)

    return i, row

def fusion_reactivity_rz(vel, xsi, zArr2D, rArr2D, f_rz, makeplot=False, savename=''):
    """
    Wrapper function to calculate the fusion reactivity / unit volume for a
    given set of distribution functions on r-z.

    Parameters
    ----------
    vel : 2D numpy array  
        Velocity grid in m/s, shape (len(vPar), len(vPerp))
    xsi : 2D numpy array
        Normalized pitch angle grid, shape (len(vPar), len(vPerp))
    zArr2D : np.array
        z-locations where the f has been calculated. [meters]
    rArr2D : np.array
        r-locations where the f has been calculated. [meters].
    f_rz : np.array
        Distribution function at all points in the plasma.
        This is a 4D array with the following index convention-
        1st index - r-position
        2nd index - z-position
        3rd index - vel
        4th index - xsi
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
    vel_1d = vel[0, :]
    xsi_1d = xsi[:, 0]

    # Empty array to store the 2D reactivity profile
    reactivity2D = np.zeros_like(zArr2D)

    # Global variables that help remove redundant computation in the parallelization workflow
    _K = None
    _vel_1d = None
    _xsi_1d = None
    _f_rz = None
    _symv = None

    # Pool initializer: runs once per worker process and stashes K and the
    # grids as module globals (see compute_row_fz's docstring for why this
    # pattern is used). Also resolves the BLAS symv function once per
    # worker here rather than once per (r,z) point inside
    # compute_row_reactivity, since scipy's BLAS function lookup has its
    # own dispatch overhead.
    def _init_worker(K, vel_1d, xsi_1d, f_rz):
        global _K, _vel_1d, _xsi_1d, _f_rz, _symv
        _K = K
        _vel_1d = vel_1d
        _xsi_1d = xsi_1d
        _f_rz = f_rz
        _symv = sc.linalg.blas.get_blas_funcs('symv', (K,))

    # Build the kernel ONCE, before any workers exist — it only depends on
    # the (vel_1d, xsi_1d) grid, which is the same for every (r,z) point,
    # so every worker can reuse this same K (see build_fusion_kernel).
    K, vel_1d_ext, xsi_1d_ext = build_fusion_kernel(vel_1d, xsi_1d)

    # Use half the lana cores to parallelize the workflow
    with Pool(int(cpu_count()/2), initializer=_init_worker, initargs=(K, vel_1d, xsi_1d, f_rz)) as pool:
        results = pool.map(compute_row_reactivity, range(zArr2D.shape[0]))

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
