# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:33:48 2021

@author: kunalsanwalka

This program contains a list of functions used to extract useful information
from the netCDF4 files output by CQL3D.

Most functions are written to be standalone, requiring only the location of the
file. However, each function has its own docstring and one should refer to that
to understand function behaviour.

Packages needed to run functions in this file-
1. numpy
2. netCDF4
3. matplotlib
4. scipy
5. csv
6. sys
"""

import sys
sys.path.insert(1,'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Python Scripts/')

import csv
import time
import numpy as np
import netCDF4 as nc
import scipy.constants as const
import matplotlib.pyplot as plt
import eqdsk_analysis_functions as eqTools
from tabulate import tabulate
from itertools import chain
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
plt.rcParams.update({'font.size': 32})

# =============================================================================
# Plot Directory
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

# =============================================================================
# Processed Data Directory
# =============================================================================
dataDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Processed Data/'

# =============================================================================
# CQL3D Data Directory
# =============================================================================
cql3dDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/CQL3D/'

#TODO
def zero_d_parameters(filename,longParams=False):
    """
    This function returns a dictionary with the values for the 0D parameters
    in a given CQL3D run. The keys for each parameter and what they mean are-
    
    ngen        = Number of general species (distribution function evaluated)
    rfPow       = Total absorbed RF Power [W] (for ngen>1, returns array)
    nbiPow      = Total absorbed NBI Power [W] (for ngen>1, returns array)
    rxRate      = Reaction rate [s^-1] at the final timestep for all 4 
                  reactions. The order of the reactions in the array that is 
                  returned is-
                  1. D + T --> n + 4He
                  2. D + 3He --> p + 4He
                  3. D + D --> n + 3He
                  4. D + D --> p + T
    
    Some parameters take a long time to process. These are only calculated when
    longParams=True. They are-
    
    maxBeta     = Maximum beta in the plasma
    
    If the parameter is not present in the output, the function returns an
    array of zeroes.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    longParams : boolean
        Calculate 0D values that take a longer time to compute

    Returns
    -------
    zeroDParams : Dictionary
        Dictionary with the 0D parameters
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Radially integrated RF Power
    try:
        sorpw_rfi=np.array(ds['sorpw_rfi'][:])
    except:
        sorpw_rfi=np.zeros((len(rya),ngen))
    
    #Radially integrated NBI Power
    try:
        sorpw_nbii=np.array(ds['sorpw_nbii'][:])
    except:
        sorpw_nbii=np.zeros((len(rya),ngen))
        
    #Fusion reactivity
    fusrxrt,time=fusion_rx_rate(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Create the dictionary
    zeroDParams={}
    
    #Number of general species
    zeroDParams['ngen']=ngen
    
    #Total RF Power absorbed
    zeroDParams['rfPow']=sorpw_rfi[-1]
    
    #Total NBI Power absorbed
    zeroDParams['nbiPow']=sorpw_nbii[-1]
    
    #Reaction rate
    zeroDParams['rxRate']=fusrxrt[:,-1]
    
    #Calculate the parameters that take a long time to compute
    if longParams==True:
        
        #Get the beta map
        beta_z,solrz,solzz=beta(filename)
        
        #Maximum beta
        zeroDParams['maxBeta']=np.max(beta_z)
    
    return zeroDParams

def species_labels(filename):
    """
    This function generates an array with the labels for each 'general' species
    That is, a species whos distribution function has been explicitly 
    calculated by CQL3D.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    speciesLabels : array
        Labels of all the general species.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Name of each species and specification (general, maxwellian etc.)
    kspeci=np.array(ds['kspeci'][:])
    
    # =========================================================================
    # Generate the species labels
    # =========================================================================
    
    #Array to store species labels
    speciesLabels=[]
    
    #Get a slice of kspeci which just has the label and type
    kspeciSlice=kspeci[:,:,0]
    #Append correct names to the labelling array
    for i in range(len(kspeciSlice)):
        if kspeciSlice[i,1]==b'g':
            
            #Label in CQL3D
            cqlLabel=kspeciSlice[i,0].decode('utf-8')
            
            #Come up with a nicer label
            niceLabel=''
            if cqlLabel=='d' or cqlLabel=='D' or cqlLabel=='Deuterium' or cqlLabel=='deuterium':
                niceLabel='D'
            elif cqlLabel=='t' or cqlLabel=='T' or cqlLabel=='Tritium' or cqlLabel=='tritium':
                niceLabel='T'
                
            #Add it to the array
            speciesLabels.append(niceLabel)
    
    return speciesLabels

def dist_func(filename,makeplot=False,saveplot=False,fluxsurfplot=0,species=0,vMax=8e6):
    """
    This function returns the plasma distribution function along with the
    associated velocity coordinate arrays. If there are multiple general
    species in the CQL3D run, the 'species' keyword picks the species being
    plotted and returned by this function.
    
    The distribution function as output by CQL3D is of the form-
    f(rdim,xdim,ydim)
    
    Here-
    rdim = Flux surface
    xdim = Momentum-per-mass (=x)
    ydim = Pitch angle (=y)
    
    To convert this into f(vPar,vPerp) we need to construct the vPar and vPerp
    arrays. This is done by decomposing the momentum (~velocity) into its
    parallel and perpendicular components based on the pitch angle.
    
    All three output arrays are 3D with the first index used to indicate the
    flux surface.
    
    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    fluxsurfplot : int
        Flux surface number to be plotted (0=innermost flux surface).
    species : int
        Index of species.
    vMax : float
        Maximum value of the velocity in the plots [m/s].

    Returns
    -------
    f : np.array
        Distribution function. 
        Units- 1/(m^3*(m/s)^3)
        It has the form- f(flux surface,v_par,v_perp)
    vPar : np.array
        Parallel velocity (with respect to the magnetic field).
        It has the form- vPar(flux surface,pitch angle,momentum-per-mass)
        Units- m/s
    vPerp : np.array
        Perpendicular velocity (with respect to the magnetic field).
        It has the form- vPerp(flux surface,pitch angle,momentum-per-mass)
        Units- m/s
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])

    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #Velocity normalization factor
    vnorm=float(ds['vnorm'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Create the vperp and vpar arrays
    # =========================================================================
    
    #Create the variables
    vPerp=np.zeros((lrz,iy,jx))
    vPar=np.zeros((lrz,iy,jx))
    
    #Calculate the velocities based on the magnitude and pitch angle for each
    #flux surface
    for k in range(0,lrz):
        for i in range(0,iy):
            for j in range(0,jx):
                vPerp[k,j,i]=vnorm*x[j]*np.sin(y[k,i])
                vPar[k,j,i]=vnorm*x[j]*np.cos(y[k,i])
    
    #Convert velocity to m/s from cm/s
    vPar/=100
    vPerp/=100
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 1e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    #Convert from CQL3D to SI units
    f*=1e12/(vnorm**3)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_dist_func_fluxsurf_'+str(fluxsurfplot)+'.png'
        if multiSpecies:
            savename=ncName+'_dist_func_fluxsurf_'+str(fluxsurfplot)+'_species_'+speciesLabels[species]+'.png'
        
        #Represent the r/a value in scientific notation
        ryaSciNot="{:.2e}".format(rya[fluxsurfplot])
        
        #Convert data to log
        logData=np.log10(f[fluxsurfplot])
        
        #Maximum of the distribution
        maxDist=np.round(np.max(logData))
        maxDist=2
        minDist=maxDist-15
        
        #Create the plot
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        
        pltobj=ax.contourf(vPar[fluxsurfplot],vPerp[fluxsurfplot],logData,levels=np.linspace(minDist,maxDist,31))
        
        ax.set_xlabel(r'$v_{||}$ [m/s]')
        ax.set_xlim(-vMax,vMax)
        ax.set_xticks(np.linspace(-vMax,vMax,11))
        
        ax.set_ylabel(r'$v_{\perp}$ [m/s]')
        ax.set_ylim(0,vMax)
        ax.set_yticks(np.linspace(0,vMax,6))
        
        ax.set_title('Distribution Function (r/a = '+ryaSciNot+')')
        ax.contour(pltobj,colors='black')
        ax.grid(True)
        
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'log$_{10}$(v$^{-3}$)')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return f,vPar,vPerp

def dist_func_z_evol(filename, species=0):
    """
    This function calculates the evolved distribution function along z by
    conserving the 1st adiabatic invariant mu.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    species : int
        Index of species.

    Returns
    -------
    f : np.array
        Distribution function. 
        Units- 1/(m^3*(m/s)^3)
        It has the form- f(flux surface, zIndex, v_par, v_perp)
    vPar : np.array
        Parallel velocity (with respect to the magnetic field).
        It has the form- vPar(flux surface,pitch angle,momentum-per-mass)
        Units- m/s
    vPerp : np.array
        Perpendicular velocity (with respect to the magnetic field).
        It has the form- vPerp(flux surface,pitch angle,momentum-per-mass)
        Units- m/s
    solrz : np.array
        DESCRIPTION.
    solzz : np.array
        DESCRIPTION.
    """
    
    #Open the file
    ds = nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi = np.array(ds['bbpsi'][:])

    #R positions [cm]
    solrz = np.array(ds['solrz'][:])

    #Z positions [cm]
    solzz = np.array(ds['solzz'][:])
    
    #Normalized radius
    rya = np.array(ds['rya'][:])
    
    #Get the distribution function data
    distData, vPar, vPerp = dist_func(filename, species=species)

    # =========================================================================
    # Put the data on a regular grid
    # =========================================================================

    #Define new variables
    distDataNew = []

    #Regular grid on which to interpolate the data
    vParMax = np.max(vPar)
    vPerpMax = np.max(vPerp)
    vParReg, vPerpReg = np.meshgrid(np.linspace(-vParMax, vParMax, 250),
                                    np.linspace(0, vPerpMax, 200))

    for surface in range(len(distData)):
        
        distDataCurr = distData[surface]
        vParCurr = vPar[surface]
        vPerpCurr = vPerp[surface]
        
        #Flatten everything to 1D
        distDataFlat = np.array(list(chain.from_iterable(distDataCurr)))
        vParFlat = np.array(list(chain.from_iterable(vParCurr)))
        vPerpFlat = np.array(list(chain.from_iterable(vPerpCurr)))
        
        #List of tuples for the gridddata function
        velPoints = np.array((vParFlat,vPerpFlat)).T
        
        #Put the data on a regular grid
        distDataRegular = griddata(velPoints, distDataFlat, (vParReg,vPerpReg))
        
        distDataNew.append(distDataRegular)
        
    #Convert to numpy
    distDataNew = np.array(distDataNew)
    #Set all nan values to 0
    distDataNew = np.nan_to_num(distDataNew)

    #Rename everything to the original names
    distData = distDataNew
    vPar = vParReg
    vPerp = vPerpReg
    
    # =========================================================================
    # Evolve the distribution functions in z based on mu conservation
    # =========================================================================
    
    #Array to store the z evolved distribution functions
    #Index arrangement-
    #0- Flux surface
    #1- Z index
    #2- vPerp
    #3- vPar
    distDataZEvol=np.zeros((np.shape(distData)[0],
                            np.shape(solzz)[1],
                            np.shape(vPar)[0],
                            np.shape(vPar)[1]))

    #Go over each flux surface
    for surface in range(len(distData)):
    # for surface in range(2):
        
        print('Flux surface- '+str(surface))
        
        #Go over each z position
        for zInd in range(np.shape(solzz)[1]):
        # for zInd in range(2):
            
            print('z position [cm]- '+str(solzz[0,zInd]))
            
            #Evolve the vPerp array
            vPerpNew = vPerp*np.sqrt(bbpsi[surface,zInd])
            
            #Evolve the vPar array
            vParNew = np.sqrt(vPar**2 + (vPerp**2)*(1-bbpsi[surface,zInd]))
            
            #We removed all the negative vPar values when we squared and square-
            #rooted. Manually adding them back.
            vParPositive = vParNew[:,int(len(vPar[0])/2):]
            vParNegative = -vParNew[:,:int(len(vPar[0])/2)]
            
            vParNew[:,int(len(vPar[0])/2):] = vParPositive
            vParNew[:,:int(len(vPar[0])/2)] = vParNegative
            
            #Flatten everything to 1D
            distDataFlat = list(chain.from_iterable(distData[surface]))
            vPerpFlat = list(chain.from_iterable(vPerpNew))
            vParFlat = list(chain.from_iterable(vParNew))
            
            #Convert to numpy
            distDataFlat = np.array(distDataFlat)
            vPerpFlat = np.array(vPerpFlat)
            vParFlat = np.array(vParFlat)
            
            #Remove all entries where vPar==nan
            distDataFlat = distDataFlat[~np.isnan(vParFlat)]
            vPerpFlat = vPerpFlat[~np.isnan(vParFlat)]
            vParFlat = vParFlat[~np.isnan(vParFlat)]
            
            #List of tuples for the gridddata function
            velPoints = np.array((vParFlat,vPerpFlat)).T
            
            #Put the data on a regular grid
            distDataRegular = griddata(velPoints, distDataFlat, (vPar,vPerp))
            
            #Append to the array
            distDataZEvol[surface,zInd] = distDataRegular
            
    #Set all nan values to 0
    distDataZEvol = np.nan_to_num(distDataZEvol)

    return distDataZEvol, vPar, vPerp, solrz, solzz

def dist_func_energy(filename, makeplot=False, saveplot=False, 
                     speciesLabel=0):
    """
    This function returns the distribution function as a function of the energy
    only.
    
    The return arrays are 2D with the form-
    returnArr[flux surface, energy value]

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    species : int
        Index of species.

    Returns
    -------
    distFunc1D : np.array
        1D distribution function per flux surface.
    energyArr1D : np.array
        Associated energy value in eV.
    """
    
    #TODO- Accurately account for the jacobian when integrating 1 velocity axis
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Get the distribution function data
    distData, vPar, vPerp = dist_func(filename,
                                      species=speciesLabel)
    
    # =========================================================================
    # Computation
    # =========================================================================
    
    #Array to store the results per flux surface
    energyArr1D = np.zeros(shape=(len(distData), len(distData[0])))
    distFunc1D = np.zeros(shape=(len(distData), len(distData[0])))
    
    
    #Go over each flux surface
    for j in range(len(distData)):
        
        #Get the data for that flux surface
        distDataCurr = distData[j]
        vParCurr = vPar[j]
        vPerpCurr = vPerp[j]
        
        #Calculate the energy axis
        energyArr = 0.5 * (2*const.m_p) * (vParCurr**2 + vPerpCurr**2)
        #Convert to eV
        energyArr /= const.e
        
        energyArr1D[j] = energyArr[:, 0]
        
        #Go over each energy bin and add the data to distFunc1D
        for i in range(len(energyArr[0, :])):
            
            distFunc1D[j, i] = np.sum(distDataCurr[i, :])
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot == True:
        
        colorArr = np.linspace(0, 1, len(distData))
        
        #Convert the data to log
        logData = np.log10(distFunc1D)
        
        #Maximum of the distribution
        maxDist=np.round(np.max(logData)) + 1
        minDist=maxDist-15
        
        fig, ax = plt.subplots(figsize=(12,8))
        
        #Go over each flux surface
        for i in range(len(distData)):
            
            #Plot label
            currLabel = str(np.round(rya[i],2))
            
            if i == 0:
                
                currLabel = 'r/a = '+currLabel
                
                ax.plot(energyArr1D[i]/1e3, logData[i], linewidth=5,
                        color=[colorArr[i], 0, 0], label=currLabel)
             
            else:
                
                ax.plot(energyArr1D[i]/1e3, logData[i], linewidth=5,
                        color=[colorArr[i], 0, 0], label=currLabel)
        
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel(r'Distribution Function [log$_{10}$(E)]')
        ax.set_ylim(minDist, maxDist)
        ax.set_xlim(0, 40)
        
        ax.legend(loc=(1.01, 0))
        ax.grid()
        
        if saveplot == True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_distfunc_vs_energy.png'
        
        plt.show()
    
    return distFunc1D, energyArr1D
    
def dist_func_derivatives(filename,makeplot=False,saveplot=False,fluxsurfplot=0,species=0):
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Processed distribution function
    f,vPar,vPerp=dist_func(filename,species=species)
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
    
    # =========================================================================
    # Process the data
    # =========================================================================
    
    #Number of flux surfaces
    fluxSurfs=len(f)
    
    #Array to store the derivatives
    dfdvperp=[]
    
    #Go over each flux surface
    for i in range(0,fluxSurfs):
        
        #Take the derivative
        dfdvperpCurr=np.gradient(f[i],vPerp[i],axis=0)
        
        #Append to the array
        dfdvperp.append(dfdvperpCurr)
        
    #Convert to numpy array
    dfdvperp=np.array(dfdvperp)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_dfdvperp_fluxsurf_'+str(fluxsurfplot)+'.png'
        if multiSpecies:
            savename=ncName+'_dfdvperp_fluxsurf_'+str(fluxsurfplot)+'_species_'+speciesLabels[species]+'.png'
        
        #Represent the r/a value in scientific notation
        ryaSciNot="{:.2e}".format(rya[fluxsurfplot])
        
        #Convert data to log
        logData=np.log10(f[fluxsurfplot])
        
        #Maximum of the distribution
        maxDist=np.round(np.max(logData))
        minDist=maxDist-15
        
        #Create the plot
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(vPar[fluxsurfplot],vPerp[fluxsurfplot],logData,levels=np.linspace(minDist,maxDist,31))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel(r'$v_{||}$ [m/s]')
        ax.set_xlim(-8e6,8e6)
        ax.set_xticks(np.linspace(-8e6,8e6,17))
        ax.set_ylabel(r'$v_{\perp}$ [m/s]')
        ax.set_ylim(0,8e6)
        ax.set_title(r'$df/dv_{perp}$ (r/a = '+ryaSciNot+')')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'log$_{10}$(v$^{-3}$)')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return

def ion_dens(filename,makeplot=False,saveplot=False,savedata=False,efastd=6,species=0):
    """
    This function returns the ion densities along with the associated 
    coordinate arrays.
    
    CQL3D does not output the ion densities directly. They are calculated by
    taking an integral of the distribution function over velocity space.
    
    Here, the ion densities are defined by-
    
    Fast ions = >6keV
    Warm ions = <6keV
    Total ions = Warm ions + Fast ions
    
    This threshold can be changed by altering the 'efastd' variable.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    savedata : boolean
        Save the data.
    efastd : float
        Boundary between warm and fast ions (keV).
    species : int
        Index of species.

    Returns
    -------
    ndwarmz : np.array
        Warm ion density function.
        It has the form- ndwarmz(radial position,z position)
        Units - m^-3
    ndfz : np.array
        Fast ion density function.
        It has the form- ndfz(radial position,z position)
        Units - m^-3
    ndtotz : np.array
        Total ion density function.
        It has the form- ndtotz(radial position,z position)
        Units - m^-3
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
        Units - m
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
        Units - m
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Major radius of z points (=r)
    solrz=np.array(ds['solrz'][:])
    
    #Height of z points (=z)
    solzz=np.array(ds['solzz'][:])
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])
    
    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #dx centered on x-mesh points
    dx=np.array(ds['dx'][:])
    
    #Velocity normalization factor
    vnorm=float(ds['vnorm'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 1e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Create the ion density arrays
    # =========================================================================
    
    #Pitch angles from one central radial point
    pitchAngleArr=y[0,:]
    
    #Pitch angle step size
    dtheta=np.max(pitchAngleArr)/len(pitchAngleArr)
    dthetad2=0.5*dtheta
    
    #Create the theta arrays
    theta0=np.zeros(iy) #Uniformly spaced theta array
    ctheta=np.zeros(iy) #cos of theta0
    stheta=np.zeros(iy) #sin of theta0
    #TODO- Figure out what these are for. They look like various integrands
    theta1=np.zeros(iy)
    theta2=np.zeros(iy)
    theta3=np.zeros(iy)
    
    #Define the values of the theta arrays
    for i in range(0,iy):
        theta0[i]=dthetad2+i*dtheta
        stheta[i]=np.sin(theta0[i])
        ctheta[i]=np.cos(theta0[i])
        theta1[i]=2*np.pi*stheta[i]*dtheta
        theta2[i]=theta1[i]*(ctheta[i]**2)
        theta3[i]=np.pi*(stheta[i]**3)*dtheta
        
    #Create the x location arrays
    #TODO- Figure out what these are for. They look like various integrands
    xloc1=np.zeros(jx)
    xloc2=np.zeros(jx)
    
    #Define the xloc arrays
    for i in range(0,jx):
        xloc1[i]=(x[i]**2)*dx[i]
        xloc2[i]=(vnorm**2)*(x[i]**2)*xloc1[i]
        
    #Create the cosz and sinz arrays
    cosz=np.zeros((iy,lz,lrz))
    bsinz=np.zeros((iy,lz,lrz))
    
    #Define cosz and sinz
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                sign=-1    
                if y[ilr,i]<=(np.pi/2):
                    sign=1.0
                else:
                    sign=-1.0
                if (1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)>0:
                    cosz[i,ilz,ilr]=sign*np.sqrt(1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)
                bsinz[i,ilz,ilr]=np.sqrt(1-cosz[i,ilz,ilr]**2)
    
    #Create itheta
    #TODO- What is itheta?
    itheta=np.zeros((iy,lz,lrz))
    
    #Define itheta
    for lr in range(0,lrz):
        for l in range(0,lz):
            for i in range(0,int(iy/2)):
                tempvalArr=np.where(bsinz[0:int(iy/2),l,lr]>=stheta[i])
                if np.size(tempvalArr)==0:
                    tempval=0
                else:
                    tempval=np.min(tempvalArr)
                #Check if tempval is larger than iy/2
                if tempval>(iy/2):
                    itheta[i,l,lr]=int(iy/2)
                else:
                    itheta[i,l,lr]=tempval
                #Make itheta symmetric
                itheta[iy-i-1,l,lr]=itheta[i,l,lr]
    
    #Create the ion density arrays
    ndfz=np.zeros((lz,lrz)) #Fast ions
    ndwarmz=np.zeros((lz,lrz)) #Warm ions
    ndtotz=np.zeros((lz,lrz)) #Total ions
    
    #Species atomic number (assume D for single species)
    anumd=0
    if len(speciesLabels)==1:
        anumd=2
    else: #multi-ion cql3d simulation
        if speciesLabels[species]=='D':
            anumd=2
        elif speciesLabels[species]=='T':
            anumd=3
        
    #Velocity of the fast ions
    vfastd=np.sqrt(2*efastd*1000/(anumd*938e6))*3e10
    #Array with indices where velocity is greater than vfastd
    fastArr=np.where(vnorm*x>=vfastd)
    #Minimum index
    jfast_mind=np.min(fastArr)
    
    #Calculate the ion densities
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for ij in range(0,jx): #energy bins
                ithetahere=itheta[:,ilz,ilr].astype(int)
                
                #Total ion density
                ndtotz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
                
                #Fast ion density
                if ij>=jfast_mind:
                    ndfz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
                    
                #Warm ion density
                else:
                    ndwarmz[ilz,ilr]+=np.sum(theta1*xloc1[ij]*f[ilr,ij,ithetahere])
    
    #Transpose the density arrays to match the indexing convention of Python. 
    #ndfz,ndwarmz and ndtotz use the same indexing convention as IDL. Since 
    #solrz and solzz follow the Python convention, we need to make sure the 
    #density arrays are consistent with solrz and solzz
    ndfz=np.transpose(ndfz)
    ndtotz=np.transpose(ndtotz)
    ndwarmz=np.transpose(ndwarmz)
    
    #Convert solrz and solzz from cm to m
    solrz /= 100
    solzz /= 100
    
    #Convert density from 1/cm^3 to 1/m^3
    ndwarmz *= 1e6
    ndfz *= 1e6
    ndtotz *= 1e6
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot == True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName = filename.split('/')[-1]
        #Remove the .nc part
        ncName = ncName[0:-3]
        #Add suffix for all the plots
        savenameFast = ncName+'_fast_ion_dens.png'
        savenameTot = ncName+'_total_ion_dens.png'
        savenameWarm = ncName+'_warm_ion_dens.png'
        if multiSpecies:
            savenameFast = ncName+'_fast_ion_dens_species_'+speciesLabels[species]+'.png'
            savenameTot = ncName+'_total_ion_dens_species_'+speciesLabels[species]+'.png'
            savenameWarm = ncName+'_warm_ion_dens_species_'+speciesLabels[species]+'.png'
        
        #Normalize all plots with respect to each other
        maxDens = np.max(ndtotz) #m^-3
        #Round maxDens to the nearest 5e19 for plot colorbar
        maxDensRounded = np.round(maxDens/5e19) * 5e19
        
        # =====================================================================
        # Fast ion density
        # =====================================================================
        
        fig1 = plt.figure(figsize=(20, 8))
        ax1 = fig1.add_subplot(111)
        
        pltobj = ax1.contourf(solzz, solrz, ndfz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax1.contour(pltobj,colors='black')
        cbar1 = fig1.colorbar(pltobj)
        cbar1.set_label(r'Density [m$^{-3}$]')
        cbar1.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax1.set_xlabel('Z [m]')
        ax1.set_ylabel('R [m]')
        ax1.set_title('Fast Ion Density (>'+str(efastd)+'keV); Species - '+speciesLabels[species])
        
        ax1.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameFast, bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Warm ion density
        # =====================================================================
        
        fig2 = plt.figure(figsize=(20, 8))
        ax2 = fig2.add_subplot(111)
        
        pltobj = ax2.contourf(solzz, solrz, ndwarmz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax2.contour(pltobj,colors='black')
        cbar2 = fig2.colorbar(pltobj)
        cbar2.set_label(r'Density [m$^{-3}$]')
        cbar2.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax2.set_xlabel('Z [m]')
        ax2.set_ylabel('R [m]')
        ax2.set_title('Warm Ion Density (<'+str(efastd)+'keV); Species - '+speciesLabels[species])
        
        ax2.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameWarm, bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total ion density
        # =====================================================================
        
        fig3 = plt.figure(figsize=(20,8))
        ax3 = fig3.add_subplot(111)
        
        pltobj = ax3.contourf(solzz, solrz, ndtotz,
                              levels=np.linspace(0, maxDens, 200))
        
        # ax3.contour(pltobj,colors='black')
        cbar3 = fig3.colorbar(pltobj)
        cbar3.set_label(r'Density [m$^{-3}$]')
        cbar3.set_ticks(np.linspace(0, maxDensRounded, 6))
        
        ax3.set_xlabel('Z [m]')
        ax3.set_ylabel('R [m]')
        ax3.set_title('Total Ion Density; Species - '+speciesLabels[species])
        
        ax3.grid(True)
        
        if saveplot == True:
            plt.savefig(plotDest+savenameTot, bbox_inches='tight')
        plt.show()
        
    # =========================================================================
    # Save the data
    # =========================================================================
    
    if savedata == True:
        
        #Generate the savename of the data
        #Get the name of the .nc file
        ncName = filename.split('/')[-1]
        #Remove the .nc part
        ncName = ncName[0:-3]
        savenameWarm = ncName+'_warm_ion_density.npy'
        savenameFast = ncName+'_fast_ion_density.npy'
        savenameTot = ncName+'_total_ion_density.npy'
        savenameR = ncName+'_r_grid.npy'
        savenameZ = ncName+'_z_grid.npy'
        
        np.save(dataDest+savenameWarm, ndwarmz)
        np.save(dataDest+savenameFast, ndfz)
        np.save(dataDest+savenameTot, ndtotz)
        np.save(dataDest+savenameR, solrz)
        np.save(dataDest+savenameZ, solzz)
    
    return ndwarmz, ndfz, ndtotz, solrz, solzz

def warm_dens(filename,efastd=6,species=0):
    """
    This function returns the total number of warm ions and the warm ion
    density.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    efastd : float
        Warm ion energy limit (keV).
    species : int
        Index of species.

    Returns
    -------
    fSum : float
        Total number of warm ions in the plasma (# of particles).
    warmDens : float
        Overall warm plasma density.
    """
    
    #TODO- fix warm density calculator. maybe there is a jacobian for the velocity
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Get the species labels
    speciesLabels=species_labels(filename)
    
    #Get the distribution function data
    f,vPar,vPerp=dist_func(filename,species=species)
    
    #Get the volume per flux surface
    dvol=np.array(ds['dvol'][:])
    
    #Convert from keV to J
    energyLim=efastd*100*const.e
    
    #Get the velocity limit based on the energy limit
    maxVel=0
    if speciesLabels[species]=='D':
        maxVel=np.sqrt(2*energyLim/(2*const.m_p))
    if speciesLabels[species]=='T':
        maxVel=np.sqrt(2*energyLim/(3*const.m_p))
    
    #Create linear arrays for vPar and vPerp
    vPar1DReg=np.linspace(np.max(vPar),np.min(vPar),np.shape(vPar)[1])
    vPerp1DReg=np.linspace(np.max(vPerp),np.min(vPerp),np.shape(vPerp)[1])
    #Mesh
    vParReg,vPerpReg=np.meshgrid(vPar1DReg,vPerp1DReg)
    
    #Get the magnitude of the velocity
    vMag=np.sqrt(vParReg**2+vPerpReg**2)
    
    #Total cold plasma particle count
    fSum=0
    
    #Density per flux surface
    warmDensArr=[]
    
    #Go over each flux surface
    for i in range(len(f)):
    
        currf=f[i]
        currVPar=vPar[i]
        currVPerp=vPerp[i]
        
        #Flatten the data
        currf=currf.flatten()
        currVPar=currVPar.flatten()
        currVPerp=currVPerp.flatten()
        
        #Create a 1D array of all the points
        points=np.squeeze(np.dstack((currVPar,currVPerp)))
        
        #Grid the data
        fReg=griddata(points,currf,(vParReg,vPerpReg),method='linear')
        
        #Remove all nan values (set them to 0)
        fReg=np.nan_to_num(fReg)
    
        #Define the cold ion population
        fCold=np.zeros_like(fReg)
        
        #Only consider ions below a certain velocity
        fCold[vMag<maxVel]=fReg[vMag<maxVel]
        
        ##############
        #Integrate over v_||
        fIntvPar=np.trapz(fCold,x=vPar1DReg)
        
        # return fCold,vPar1DReg,vPerp1DReg,fIntvPar,i
        
        #Integrate over v_perp
        warmDens=np.trapz(fIntvPar*2*np.pi*vPerp1DReg,x=vPerp1DReg)
        
        # print(warmDens)
        ##############
        
        #Number of cold ions in the flux surface
        coldCount=np.sum(fCold)
        
        #Add to the total warm ion count
        fSum+=coldCount
        
        #Warm ion density in the given flux surface
        warmDensArr.append(coldCount/dvol[i])
        
    #Convert to numpy
    warmDensArr=np.array(warmDensArr)
    
    #Volume averaged warm ion density
    warmDens=np.sum(warmDensArr*dvol)/np.sum(dvol)
    
    #Convert to m^{-3}
    warmDens*=1e6
    
    return warmDens
    # return fSum,warmDens

def fusion_power_dens(filename, species=0, makeplot=False, saveplot=False):
    
    # =========================================================================
    # Get the raw data
    # =========================================================================

    #Get the distribution function at all the rz positions
    distDataZEvol, vPar, vPerp, solrz, solzz = dist_func_z_evol(filename, species=species)
    
    # =========================================================================
    # Get the fusion cross section data
    # =========================================================================
    #Code in this block is from https://scipython.com/blog/plotting-nuclear-fusion-cross-sections/
    
    #Cross sections data directory
    crossSectionsDir = 'C:/Users/kunal/OneDrive - UW-Madison/WHAM/Data/Fusion Cross Sections/'
    
    # To plot using centre-of-mass energies instead of lab-fixed energies, set True
    COFM = True
    
    # Reactant masses in atomic mass units (u).
    masses = {'D': 2.014, 'T': 3.016, '3He': 3.016}
    
    # Energy grid [eV]
    energyGrid = np.arange(1, 50e3, 10)
    
    def read_xsec(filenameCX):
        """Read in cross section from filename and interpolate to energy grid."""
        
        E, xs = np.genfromtxt(filenameCX, comments='#', skip_footer=2, unpack=True)
        
        #Remove all the directory information from the filename
        filenameCX = filenameCX.split('/')[-1]
        
        if COFM:
            
            collider, target = filenameCX.split('_')[:2]
            m1, m2 = masses[target], masses[collider]
            E *= m1 / (m1 + m2)
    
        xs = np.interp(energyGrid, E*1.e3, xs*1.e-28)
        return xs
    
    # D + D -> T + p
    DDa_xs = read_xsec(crossSectionsDir + 'D_D_-_T_p.txt')
    # D + D -> 3He + n
    DDb_xs = read_xsec(crossSectionsDir + 'D_D_-_3He_n.txt')
    # Total D + D fusion cross section is due to equal contributions from the
    # above two processes.
    DD_xs = DDa_xs + DDb_xs
    
    # =========================================================================
    
    #Array to store the fusion power densities
    #Index arrangeement
    #0 - radius
    #1 - z value
    fusPowerDensArr = np.zeros(shape=(len(solzz),
                                      len(solzz[0])))
    
    #Go over each flux surface
    for i in range(len(solzz)):
        
        #Go over each z position
        for j in range(len(solzz[0])):
            
            #TODO - Complete this function
            i=1
    
    return

def radial_density_profile(filename,makeplot=False,saveplot=False):
    """
    This function calculates the radial density profile for all the general
    species at z=0. The densities have units of m^{-3}.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    radArr : np.array
        Radial coordinates where the density is calculated [m].
    radDens : np.array
        Radial density in m^{-3}. If there is more than 1 general species, the
        1st index in this array is used to mark the species.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Get the species labels
    speciesLabels=species_labels(filename)
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Array to store the radial coordinates
    radArr=[]
    
    #Array to store the radial density profile
    radDens=[]
    
    #Go over each species
    for i in range(ngen):
        
        #Get the ion density data
        ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,species=i)
        
        #Get the radial coordinates
        radArr=solrz[:,0]
        
        #Get the radial density
        radDensSpecies=ndtotz[:,0]
        
        #Append to the radDens array
        radDens.append(radDensSpecies)
        
    #Convert to numpy array
    radDens=np.array(radDens)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_title('Radial Density Profile')
        ax.set_xlabel('Radius [m]')
        ax.set_ylabel(r'Density [m$^{-3}$]')
        
        #Go over each species
        if ngen>=2:
            for i in range(ngen):
                ax.plot(radArr,radDens[i],label=speciesLabels[i],color=colors[i],linewidth=5)
        else:
            ax.plot(radArr,radDens[0],label=speciesLabels[0],color=colors[0],linewidth=5)
            
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_radial_density_profile.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return radArr,radDens[0]

def axial_density_profile(filename,makeplot=False,saveplot=False):
    """
    This function calculates the axial density profile for all the general
    species at the innermost flux surface. The densities have units of m^{-3}.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    axialArr : np.array
        Axial coordinates where the density is calculated [m].
    axialDens : np.array
        Axial density in m^{-3}. If there is more than 1 general species, the
        1st index in this array is used to mark the species.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Get the species labels
    speciesLabels=species_labels(filename)
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Array to store the axial coordinates
    axialArr=[]
    
    #Array to store the axial density profile
    axialDens=[]
    
    #Go over each species
    for i in range(ngen):
        
        #Get the ion density data
        ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,species=i)
        
        #Get the axial coordinates
        axialArr=solzz[0]
        
        #Get the axial density
        axialDensSpecies=ndtotz[0]
        
        #Append to the radDens array
        axialDens.append(axialDensSpecies)
    
    #Convert to numpy array
    axialDens=np.array(axialDens)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_title('Axial Density Profile')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'Density [m$^{-3}$]')
        
        #Go over each species
        if ngen>=2:
            for i in range(ngen):
                ax.plot(axialArr,axialDens[i],label=speciesLabels[i],color=colors[i],linewidth=5)
        else:
            ax.plot(axialArr,axialDens[0],label=speciesLabels[0],color=colors[0],linewidth=5)
            
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_axial_density_profile.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return axialArr,axialDens[0]

def pressure(filename,makeplot=False,saveplot=False,savedata=False,species=0):
    """
    This function returns the pressures for each species along with the 
    associated coordinate arrays.
    
    CQL3D does not output the pressures directly. They are calculated by
    taking an integral of the distribution function over velocity space.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    savedata : boolean
        Save the data.
    species : int
        Index of species.

    Returns
    -------
    pressparz_d : np.array
        Parallel pressure.
        It has the form- pressparz_d(radial position,z position)
    pressprpz_d : np.array
        Perpendicular pressure.
        It has the form- pressprpz_d(radial position,z position)
    pressz_d : np.array
        Total pressure.
        It has the form- pressz_d(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Major radius of z points (=r)
    solrz=np.array(ds['solrz'][:])
    
    #Height of z points (=z)
    solzz=np.array(ds['solzz'][:])
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Distribution function
    f=np.array(ds['f'][:])
    
    #Pitch angle array
    y=np.array(ds['y'][:])
    
    #Maximum pitch angle dimension (=ydim)
    iy=int(ds['iy'][:])
    
    #Normalized momentum-per-mass array
    x=np.array(ds['x'][:])
    
    #Momentum-per-mass dimension (=xdim)
    jx=int(ds['jx'][:])
    
    #dx centered on x-mesh points
    dx=np.array(ds['dx'][:])
    
    #Velocity normalization factor
    vnorm=float(ds['vnorm'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Species Labels
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Clean up distribution function
    # =========================================================================
    
    #Remove all nan values (set them to 0)
    f=np.nan_to_num(f)
    
    #Set all values at or below 0 to 10e-5 (helps with taking the log)
    f[f<=0]=1e-5
    
    # =========================================================================
    # Check if the distribution function has multiple species
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get the f for the right species (else f has the wrong shape)
        f=f[species]
    
    # =========================================================================
    # Calculate the pressure profiles
    # =========================================================================
    
    #NOTE- The _d suffix is an artifact of single species processing code. It
    #      has no bearing on the actual species being analyzed
    
    #Pitch angles from one central radial point
    pitchAngleArr=y[0,:]
    
    #Pitch angle step size
    dtheta=np.max(pitchAngleArr)/len(pitchAngleArr)
    dthetad2=0.5*dtheta
    
    #Create the theta arrays
    theta0=np.zeros(iy) #Uniformly spaced theta array
    ctheta=np.zeros(iy) #cos of theta0
    stheta=np.zeros(iy) #sin of theta0
    #TODO- Figure out what these are for. They look like various integrands
    theta1=np.zeros(iy)
    theta2=np.zeros(iy)
    theta3=np.zeros(iy)
    
    #Define the values of the theta arrays
    for i in range(0,iy):
        theta0[i]=dthetad2+i*dtheta
        stheta[i]=np.sin(theta0[i])
        ctheta[i]=np.cos(theta0[i])
        theta1[i]=2*np.pi*stheta[i]*dtheta
        theta2[i]=theta1[i]*(ctheta[i]**2)
        theta3[i]=np.pi*(stheta[i]**3)*dtheta
        
    #Create the x location arrays
    #TODO- Figure out what these are for. They look like various integrands
    xloc1=np.zeros(jx)
    xloc2=np.zeros(jx)
    
    #Define the xloc arrays
    for i in range(0,jx):
        xloc1[i]=(x[i]**2)*dx[i]
        xloc2[i]=(vnorm**2)*(x[i]**2)*xloc1[i]
        
    #Create the cosz and sinz arrays
    cosz=np.zeros((iy,lz,lrz))
    bsinz=np.zeros((iy,lz,lrz))
    
    #Define cosz and sinz
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                sign=-1    
                if y[ilr,i]<=(np.pi/2):
                    sign=1.0
                else:
                    sign=-1.0
                if (1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)>0:
                    cosz[i,ilz,ilr]=sign*np.sqrt(1-bbpsi[ilr,ilz]*np.sin(y[ilr,i])**2)
                bsinz[i,ilz,ilr]=np.sqrt(1-cosz[i,ilz,ilr]**2)
    
    #Create itheta
    #TODO- What is itheta?
    itheta=np.zeros((iy,lz,lrz))
    
    #Define itheta
    for lr in range(0,lrz):
        for l in range(0,lz):
            for i in range(0,int(iy/2)):
                tempvalArr=np.where(bsinz[0:int(iy/2),l,lr]>=stheta[i])
                if np.size(tempvalArr)==0:
                    tempval=0
                else:
                    tempval=np.min(tempvalArr)
                #Check if tempval is larger than iy/2
                if tempval>(iy/2):
                    itheta[i,l,lr]=int(iy/2)
                else:
                    itheta[i,l,lr]=tempval
                #Make itheta symmetric
                itheta[iy-i-1,l,lr]=itheta[i,l,lr]
                
    #Create the pressure profile arrays
    pressparz_d=np.zeros((lz,lrz)) #Parallel pressure
    pressprpz_d=np.zeros((lz,lrz)) #Perpendicular pressure
    
    #Mass of the species (assume D for single species)
    anumd=2
    if speciesLabels[species]=='D':
        anumd=2
    elif speciesLabels[species]=='T':
        anumd=3
    massSpec=anumd*1.67e-24
    
    #Calculate the pressure profiles
    for ilr in range(0,lrz): #flux surfaces
        for ilz in range(0,lz): #z positions
            for i in range(0,iy): #pitch angles
                ithetahere=int(itheta[i,ilz,ilr])
                pressparz_d[ilz,ilr]+=massSpec*theta2[i]*np.sum(xloc2[:]*f[ilr,:,ithetahere])
                pressprpz_d[ilz,ilr]+=massSpec*theta3[i]*np.sum(xloc2[:]*f[ilr,:,ithetahere])
                    
    #Transpose the pressure arrays to match the indexing convention of Python. 
    #pressprpz_d and pressprpz_d use the same indexing convention as IDL. Since 
    #solrz and solzz follow the Python convention, we need to make sure the 
    #pressure arrays are consistent with solrz and solzz
    pressparz_d=np.transpose(pressparz_d)
    pressprpz_d=np.transpose(pressprpz_d)
    
    #Total pressure
    pressz_d=(pressparz_d+2*pressprpz_d)/3
    
    #Convert solrz and solzz from cm to m
    solrz/=100
    solzz/=100
    
    #Convert pressures from baryes to pascals
    pressparz_d/=10
    pressprpz_d/=10
    pressz_d/=10
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure.png'
        savenamePerp=ncName+'_perp_pressure.png'
        savenameTot=ncName+'_tot_pressure.png'
        if multiSpecies:
            savenamePar=ncName+'_par_pressure_'+speciesLabels[species]+'.png'
            savenamePerp=ncName+'_perp_pressure_'+speciesLabels[species]+'.png'
            savenameTot=ncName+'_tot_pressure_'+speciesLabels[species]+'.png'
        
        #Normalize all plots with respect to each other
        maxPressure=np.max([np.max(pressz_d),np.max(pressparz_d),np.max(pressprpz_d)])
        
        # =====================================================================
        # Parallel Pressure
        # =====================================================================
        
        fig1=plt.figure(figsize=(20,8))
        ax=fig1.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressparz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Parallel Pressure')
        ax.grid(True)
        cbar1=fig1.colorbar(pltobj)
        cbar1.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePar,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Perpendicular Pressure
        # =====================================================================
        
        fig2=plt.figure(figsize=(20,8))
        ax=fig2.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressprpz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Perpendicular Pressure')
        ax.grid(True)
        cbar2=fig2.colorbar(pltobj)
        cbar2.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePerp,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total Pressure
        # =====================================================================
        
        fig3=plt.figure(figsize=(20,8))
        ax=fig3.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressz_d,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Total Pressure')
        ax.grid(True)
        cbar3=fig3.colorbar(pltobj)
        cbar3.set_label(r'Pascals [N/m$^{-2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameTot,bbox_inches='tight')
        plt.show()
        
    # =========================================================================
    # Save the data
    # =========================================================================
    
    if savedata==True:
        
        #Generate the savename of the data
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure_'+speciesLabels[species]+'.npy'
        savenamePerp=ncName+'_perp_pressure_'+speciesLabels[species]+'.npy'
        savenameTot=ncName+'_tot_pressure_'+speciesLabels[species]+'.npy'
        savenameR=ncName+'_r_grid.npy'
        savenameZ=ncName+'_z_grid.npy'
        
        np.save(dataDest+savenamePar,pressparz_d)
        np.save(dataDest+savenamePerp,pressprpz_d)
        np.save(dataDest+savenameTot,pressz_d)
        np.save(dataDest+savenameR,solrz)
        np.save(dataDest+savenameZ,solzz)
    
    return pressparz_d,pressprpz_d,pressz_d,solrz,solzz

def total_pressure(filename,makeplot=False,saveplot=False,savedata=False):
    """
    This function returns the total plasma pressure along with the associated
    coordinate arrays.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    savedata : boolean
        Save the data.

    Returns
    -------
    pressparz : np.array
        Parallel pressure.
        It has the form- pressparz_d(radial position,z position)
    pressprpz : np.array
        Perpendicular pressure.
        It has the form- pressprpz_d(radial position,z position)
    pressz : np.array
        Total pressure.
        It has the form- pressz_d(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    # =========================================================================
    # Calculate the total pressure
    # =========================================================================
    
    #Initialize the arrays
    pressparz,pressprpz,pressz,solrz,solzz=pressure(filename)
    
    #Use the standard function if there is only 1 species
    if ngen<=1:
        return pressure(filename,makeplot=makeplot,saveplot=saveplot,savedata=savedata)
    
    #Add the rest of the species
    for i in range(1,ngen):
        pressparzNew,pressprpzNew,presszNew,solrz,solzz=pressure(filename,species=i)
        pressparz+=pressparzNew
        pressprpz+=pressprpzNew
        pressz+=presszNew
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure.png'
        savenamePerp=ncName+'_perp_pressure.png'
        savenameTot=ncName+'_tot_pressure.png'
        
        #Normalize all plots with respect to each other
        maxPressure=np.max([np.max(pressz),np.max(pressparz),np.max(pressprpz)])
        
        # =====================================================================
        # Parallel Pressure
        # =====================================================================
        
        fig1=plt.figure(figsize=(20,8))
        ax=fig1.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressparz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Parallel Pressure')
        ax.grid(True)
        cbar1=fig1.colorbar(pltobj)
        cbar1.set_label(r'Pascals [N/m$^{2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePar,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Perpendicular Pressure
        # =====================================================================
        
        fig2=plt.figure(figsize=(20,8))
        ax=fig2.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressprpz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Perpendicular Pressure')
        ax.grid(True)
        cbar2=fig2.colorbar(pltobj)
        cbar2.set_label(r'Pascals [N/m$^{2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenamePerp,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # Total Pressure
        # =====================================================================
        
        fig3=plt.figure(figsize=(20,8))
        ax=fig3.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,pressz,levels=np.linspace(0,maxPressure,50))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Total Pressure')
        ax.grid(True)
        cbar3=fig3.colorbar(pltobj)
        cbar3.set_label(r'Pascals [N/m$^{2}$]')
        if saveplot==True:
            plt.savefig(plotDest+savenameTot,bbox_inches='tight')
        plt.show()
        
    # =========================================================================
    # Save the data
    # =========================================================================
    
    if savedata==True:
        
        #Generate the savename of the data
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenamePar=ncName+'_par_pressure.npy'
        savenamePerp=ncName+'_perp_pressure.npy'
        savenameTot=ncName+'_tot_pressure.npy'
        savenameR=ncName+'_r_grid.npy'
        savenameZ=ncName+'_z_grid.npy'
        
        np.save(dataDest+savenamePar,pressparz)
        np.save(dataDest+savenamePerp,pressprpz)
        np.save(dataDest+savenameTot,pressz)
        np.save(dataDest+savenameR,solrz)
        np.save(dataDest+savenameZ,solzz)
        
        print(dataDest+savenamePar)
        
    return pressparz,pressprpz,pressz,solrz,solzz

def beta(filename,makeplot=False,saveplot=False):
    """
    This function returns the plasma beta along with the associated coordinate 
    arrays.
    
    CQL3D does not output the pressures directly. They are calculated by
    taking moments of the distribution function over velocity space.
    
    NOTE: This function was originally written in Fortran, then converted
          to IDL and is finally in Python. A lot of optimizations can be made 
          to this code that are Python specific.

    Parameters
    ----------
    filename : string
        Location of the CQl3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    beta_z : np.array
        Plasma beta as a function of position.
        It has the form- beta_z(radial position,z position)
    solrz : np.array
        Radial position.
        It has the form- solrz(radial position,z position)
    solzz : np.array
        Z position.
        It has the form- solzz(radial position,z position)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Dimension of z-grid along B
    lz=int(ds['lz'][:])
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized magnetic field strength (B(z)/B(z=0))
    bbpsi=np.array(ds['bbpsi'][:])
    
    #Minimum |B| on a flux surface
    bmidplane=np.array(ds['bmidplne'][:])
    
    # =========================================================================
    # Calculate the plasma beta
    # =========================================================================
    
    #Get the pressure profiles
    pressparz_d,pressprpz_d,pressz_d,solrz,solzz=total_pressure(filename)
    
    #Transpose to match the IDL code
    pressparz_d=np.transpose(pressparz_d)
    pressprpz_d=np.transpose(pressprpz_d)
    pressz_d=np.transpose(pressz_d)
    
    #Create the beta array
    beta_z=np.zeros((lz,lrz))
    
    #Create bzz (Magnetic field strength)
    bzz=np.zeros((lz,lrz))
    
    #Calculate bzz
    for i in range(0,lrz):
        bzz[:,i]=bbpsi[i,:]*bmidplane[i]
        
    #Calculate beta_z
    beta_z=8*np.pi*pressz_d/bzz**2
    
    #Transpose the beta array to match the indexing convention of Python. 
    #beta_z use the same indexing convention as IDL. Since solrz and solzz 
    #follow the Python convention, we need to make sure the beta array is
    #consistent with solrz and solzz
    beta_z=np.transpose(beta_z)
    
    #Multiply beta by 10 because the units in the above calculation are mixed
    beta_z*=10
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_beta.png'
        
        fig=plt.figure(figsize=(20,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,beta_z,levels=50)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Plasma Beta')
        ax.grid(True)
        fig.colorbar(pltobj)
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return beta_z,solrz,solzz

def aic_growthrate(filename,makeplot=False,saveplot=False):
    
    #Get the plasma pressures
    pressparz_d,pressprpz_d,pressz_d,solrz,solzz=pressure(filename)
    
    #Get the beta
    beta_z,solrz,solzz=beta(filename)
    
    #Temperature anisotropy (same as pressure anisotropy)
    tempAniso=pressprpz_d/pressparz_d
    
    #Calculate the AIC growthrate
    gammaNorm=np.exp(-(1/beta_z)*((tempAniso-1)*(-2)))
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_aic_growthrate.png'
        
        fig=plt.figure(figsize=(20,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(solzz,solrz,gammaNorm,levels=50)
        ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Normalized AIC Growthrate')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$\gamma / \Omega_{ci}$')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return gammaNorm,solrz,solzz

def axial_neutron_flux(filename,makeplot=False,saveplot=False):
    """
    This function returns the fusion neutron flux as a function of the axial
    coordinate z.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    flux_neutron_f : np.array
        Fusion neutron flux as a function of the axial coordinate z.
        Units- W/m**2/steradian
    z_fus : np.array
        Values of z associated with flux_neutron_f.
        Units- m
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Neutron flux as a function of z
    flux_neutron_f=np.array(ds['flux_neutron_f'][:])
    
    #z array associated with the neutron flux array
    z_fus=np.array(ds['z_fus'][:])
    
    # =========================================================================
    # Convert to SI units
    # =========================================================================
    
    #Convert the z array to meters
    z_fus/=100
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_fus_flux_axial_dependence.png'
        
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(z_fus,flux_neutron_f,linewidth=5)
        ax.scatter(z_fus,flux_neutron_f,s=100)
        
        #Axes labels and sizes
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'Fusion Neutron Flux [W/m$^2$sr]')
        ax.set_xlim(min(z_fus),max(z_fus))
        
        ax.grid()
        ax.set_title('Fusion Neutron Flux')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return flux_neutron_f,z_fus

def axial_proton_flux(filename,makeplot=False,saveplot=False):
    """
    This function returns the fusion proton flux as a function of the axial
    coordinate z.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    flux_proton_f : np.array
        Fusion proton flux as a function of the axial coordinate z.
        Units- W/m**2/steradian
    z_fus : np.array
        Values of z associated with flux_proton_f.
        Units- m
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Neutron flux as a function of z
    flux_neutron_f=np.array(ds['flux_neutron_f'][:])
    
    #z array associated with the neutron flux array
    z_fus=np.array(ds['z_fus'][:])
    
    # =========================================================================
    # Calculations
    # =========================================================================
    
    #Convert neutron yield to proton yield
    flux_proton_f = flux_neutron_f * 3.02 / 2.45
    
    #Convert the z array to meters
    z_fus/=100
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_fus_proton_flux_axial_dependence.png'
        
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(z_fus,flux_proton_f,linewidth=5)
        ax.scatter(z_fus,flux_proton_f,s=100)
        
        #Axes labels and sizes
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'Fusion Proton Flux [W/m$^2$sr]')
        ax.set_xlim(min(z_fus),max(z_fus))
        
        ax.grid()
        ax.set_title('Fusion Proton Flux')
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
        plt.show()
    
    return flux_proton_f,z_fus

def radial_fusion_power(filename,makeplot=False,saveplot=False):
    """
    This function returns the total fusion power for 4 different reactions as
    function of the radius at the final timestep.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    fusPower : np.array
        Array with the various reaction power outputs. [W]
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Total fusion power for 4 different reactions
    fuspwrv=np.array(ds['fuspwrv'][:])
    
    #Volume of selected flux surfaces (cm^-3)
    dvol=np.array(ds['dvol'][:])
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Total fusion power per flux surface
    fusPower=[]
    for i in range(len(fuspwrv)):
        
        fusPower.append(fuspwrv[i]*dvol)
    
    fusPower=np.array(fusPower)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.plot(rya,fusPower[0],label=r'D + T --> n + $^4$He',linewidth=5)
        ax.plot(rya,fusPower[1],label=r'D + $^3$He --> p + $^4$He',linewidth=5)
        ax.plot(rya,fusPower[2],label=r'D + D --> n + $^3$He',linewidth=5)
        ax.plot(rya,fusPower[3],label='D + D --> p + T',linewidth=5)
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel('Fusion Power [W]')
        ax.set_title('Fusion Power')
        
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_radial_fusion_power.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,fusPower

def radial_fusion_power_density(filename,makeplot=False,saveplot=False):
    """
    This function returns the total fusion power density for 4 different 
    reactions as function of the radius at the final timestep.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
        
    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    fusPowerDens : np.array
        Array with the various reaction power density outputs. [W/m^3]
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Total fusion power for 4 different reactions
    fuspwrv=np.array(ds['fuspwrv'][:])
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Total fusion power per flux surface
    fusPowerDens=[]
    for i in range(len(fuspwrv)):
        
        fusPowerDens.append(fuspwrv[i])
    
    fusPowerDens=np.array(fusPowerDens)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.plot(rya[1:],fusPowerDens[0][1:],label=r'D + T --> n + $^4$He',linewidth=5)
        ax.plot(rya[1:],fusPowerDens[1][1:],label=r'D + $^3$He --> p + $^4$He',linewidth=5)
        ax.plot(rya[1:],fusPowerDens[2][1:],label=r'D + D --> n + $^3$He',linewidth=5)
        ax.plot(rya[1:],fusPowerDens[3][1:],label='D + D --> p + T',linewidth=5)
        
        ax.set_xlim(0,1)
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel(r'Fusion Power Density [W/cm$^3$]')
        ax.set_title('Fusion Power Density')
        
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_radial_fusion_power_dens.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,fusPowerDens

def total_fusion_power(filename):
    """
    This function returns the total fusion power for 4 different reactions at
    the final timestep.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.

    Returns
    -------
    fuspwrvt : np.array
        Array with the various reaction power outputs. [W]
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Total fusion power for 4 different reactions
    fuspwrvt=np.array(ds['fuspwrvt'][:])
    
    return fuspwrvt

def fusion_rx_rate(filename,makeplot=False,saveplot=False):
    """
    This function returns the reaction rate a function of time for 4 different
    reactions.
    
    The order of the reactions in the array that is returned is-
    1. D + T --> n + 4He
    2. D + 3He --> p + 4He
    3. D + D --> n + 3He
    4. D + D --> p + T

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    sigfft : np.array
        2D array with the reaction rates. Major axis selects for the reaction.
    time : np.array
        Corresponding time array.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Fusion reaction rate as a function of time
    sigftt=np.array(ds['sigftt'][:])
    
    #Corresponding time axis
    time=np.array(ds['time'][:])
    
    # =========================================================================
    # Transpose to allow easy selection of different reactions
    # =========================================================================
    
    sigftt=np.transpose(sigftt)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_fus_rx_rate.png'
        
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111)
        
        #Get the time in ms
        timeMs=time*1000
        
        ax.semilogy(timeMs,sigftt[0],label=r'D + T --> n + $^4$He', linewidth=5)
        ax.semilogy(timeMs,sigftt[1],label=r'D + $^3$He --> p + $^4$He', linewidth=5)
        ax.semilogy(timeMs,sigftt[2],label=r'D + D --> n + $^3$He', linewidth=5)
        ax.semilogy(timeMs,sigftt[3],label='D + D --> p + T', linewidth=5)
        
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(r'Fusion Reaction Rate [s$^{-1}$]')
        ax.set_ylim(np.max(sigftt)/1e6,np.max(sigftt)*2)
        ax.set_title('Fusion Reaction Rate')
        
        ax.grid(which='both')
        ax.legend(bbox_to_anchor=(1,1))
        
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
        plt.show()
    
    return sigftt,time

def source_power_dens(filename,makeplot=False,saveplot=False):
    """
    This function returns the amount of power deposited by each heating system
    as a function of the flux surface.
    
    If there was no NBI or RF heating in a given run, the values in the arrays
    are zeroes.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    sorpw_nbi : np.array
        NBI power density. Dimensions - (ngen,rya)
    sorpw_rf : np.array
        RF power density. Dimensions - (ngen,rya)
    sorpw_tot : np.array
        Total power density. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #General species labels
    labels=species_labels(filename)
    
    #Neutral beam power density
    try:
        sorpw_nbi=np.array(ds['sorpw_nbi'][:])*1e6 #Convert to W/m^3
    except:
        sorpw_nbi=np.zeros((len(rya),ngen))
    
    #RF power density
    try:
        sorpw_rf=np.array(ds['sorpw_rf'][:])*1e6 #Convert to W/m^3
    except:
        sorpw_rf=np.zeros((len(rya),ngen))
    
    #Total power density
    sorpw_tot=sorpw_nbi+sorpw_rf
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel(r'Power Density [W/m$^3$]')
        ax.set_title('Source Power Density')
        
        #Go over each species
        for i in range(ngen):
            
            #Plot NBI (dashed)
            ax.plot(rya,sorpw_nbi[:,i],label=labels[i]+' (NBI)',linestyle='dashed',color=colors[i])
            
            #Plot RF (dotted)
            ax.plot(rya,sorpw_rf[:,i],label=labels[i]+' (RF)',linestyle='dotted',color=colors[i])
            
            #Plot total (solid)
            ax.plot(rya,sorpw_tot[:,i],label=labels[i]+' (Total)',color=colors[i])
            
        #Make the axes look nice
        ax.set_xlim(0,1)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_source_power_density.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    #Transpose arrays to make them easier to use
    sorpw_nbi=np.transpose(sorpw_nbi)
    sorpw_rf=np.transpose(sorpw_rf)
    sorpw_tot=np.transpose(sorpw_tot)
    
    return rya,sorpw_nbi,sorpw_rf,sorpw_tot

def radial_q_profile(filename,makeplot=False,saveplot=False):
    """
    This function returns the Q_fus profile as a function of the normalized
    radius.
    
    Q_fus is calculated for each flux surface.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    qFus : np.array
        Q_fus for the given flux tube
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Volume of selected flux surfaces (cm^-3)
    dvol=np.array(ds['dvol'][:])
    
    #Total fusion power density (W/cm^3)
    fuspwrv=np.array(ds['fuspwrv'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #NBI Power Density (W/cm^3)
    try:
        sorpw_nbi=np.array(ds['sorpw_nbi'][:])
    except:
        sorpw_nbi=np.zeros((len(rya),ngen))
    
    #RF Power Density (W/cm^3)
    try:
        sorpw_rf=np.array(ds['sorpw_rf'][:])
    except:
        sorpw_rf=np.zeros((len(rya),ngen))
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Total input power density (W/cm^3)
    if len(np.shape(sorpw_nbi))==1:
        totInDens=sorpw_nbi+sorpw_rf
    else:
        totInDens=sorpw_nbi+sorpw_rf
        totInDens=np.sum(totInDens,axis=1)
    
    #Total fusion power density (W/cm^3)
    totOutDens=np.sum(fuspwrv,axis=0)
    
    #Total input power (W)
    totIn=totInDens*dvol
    
    #Total output power (W)
    totOut=totOutDens*dvol
    
    #Q_fus
    qFus=totOut/totIn
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel(r'Q$_{fus}$')
        ax.set_title('Radial Q Profile')
        
        ax.plot(rya,qFus)
        
        ax.set_xlim(0,1)
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_radial_q_profile.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,qFus

def integrated_power_density(filename,makeplot=False,saveplot=False):
    """
    This function returns the total power density deposited as a function of r 
    from both RF and NBI.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    sourceRate : np.array
        Integrated NBI power density. Dimensions - (ngen,rya)
    integratedRF : np.array
        Integrated RF power density. Dimensions - (ngen,rya)
    totalIntegrated : np.array
        Integrated total power density. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Major radius of z points (=r)
    solrz=np.array(ds['solrz'][:])
    
    #General species labels
    labels=species_labels(filename)
    
    #Source power density
    rya,sorpw_nbi,sorpw_rf,sorpw_tot=source_power_dens(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #rArr
    rArr=solrz[:,0]
    
    #Integrate over each flux surface
    sourceRate=[] #Particle source rate
    integratedRF=[]
    totalIntegrated=[]
    for i in range(len(rya)):
        
        #Trimmed arrays
        trimmedRArr=rArr[:i]
        trimmedBeamCurr=sorpw_nbi[:,:i]
        trimmedRF=sorpw_rf[:,:i]
        trimmedTot=sorpw_tot[:,:i]
        
        #Integrate over r
        sourceRate.append(np.trapz(2*np.pi*trimmedRArr*trimmedBeamCurr,trimmedRArr))
        integratedRF.append(np.trapz(2*np.pi*trimmedRArr*trimmedRF,trimmedRArr))
        totalIntegrated.append(np.trapz(2*np.pi*trimmedRArr*trimmedTot,trimmedRArr))
        
    #Convert to numpy
    sourceRate=np.array(sourceRate)
    integratedRF=np.array(integratedRF)
    totalIntegrated=np.array(totalIntegrated)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel(r'Integrated Power Density [W]')
        ax.set_title('Integrated Power Density')
        
        #Go over each species
        for i in range(ngen):
            
            #Plot NBI (dashed)
            ax.plot(rya,sourceRate[:,i],label=labels[i]+' (NBI)',linestyle='dashed',color=colors[i])
            
            #Plot RF (dotted)
            ax.plot(rya,integratedRF[:,i],label=labels[i]+' (RF)',linestyle='dotted',color=colors[i])
            
            #Plot total (solid)
            ax.plot(rya,totalIntegrated[:,i],label=labels[i]+' (Total)',color=colors[i])
            
        ax.set_xlim(0,1)
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_integrated_power_density.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
        
    #Transpose arrays to make them easier to use
    sourceRate=np.transpose(sourceRate)
    integratedRF=np.transpose(integratedRF)
        
    return rya,sourceRate,integratedRF,totalIntegrated

def radial_input_power(filename,makeplot=False,saveplot=False):
    """
    This function returns the power deposited per flux surface from both RF and
    NBI.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    
    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    integratedNBI : np.array
        NBI power [W]. Dimensions - (ngen,rya)
    integratedRF : np.array
        RF power [W]. Dimensions - (ngen,rya)
    integratedTot : np.array
        total power [W]. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #General species labels
    labels=species_labels(filename)
    
    #Volume of selected flux surfaces (cm^-3)
    dvol=np.array(ds['dvol'][:])
    
    #NBI Power Density (W/cm^3)
    try:
        sorpw_nbi=np.array(ds['sorpw_nbi'][:])
    except:
        sorpw_nbi=np.zeros((len(rya),ngen))
    
    #RF Power Density (W/cm^3)
    try:
        sorpw_rf=np.array(ds['sorpw_rf'][:])
    except:
        sorpw_rf=np.zeros((len(rya),ngen))
        
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Total input power density (W/cm^3)
    totPowDens=sorpw_nbi+sorpw_rf
    
    #Find the power in each flux surface
    rfPower=[]
    nbiPower=[]
    totalPower=[]
    for i in range(ngen):
        
        #Mutiply the power density by the volume of each flux surface
        rfPower.append(sorpw_rf[:,i]*dvol)
        nbiPower.append(sorpw_nbi[:,i]*dvol)
        totalPower.append(totPowDens[:,i]*dvol)
    
    #Convert to numpy
    rfPower=np.array(rfPower)
    nbiPower=np.array(nbiPower)
    totalPower=np.array(totalPower)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_title('Input power per flux surface')
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel('Input Power [kW]')
        
        #Go over each species
        for i in range(ngen):
            
            #Plot NBI (dashed)
            ax.plot(rya,nbiPower[i,:]/1000,label=labels[i]+' (NBI)',linestyle='dashed',color=colors[i])
            
            #Plot RF (dotted)
            ax.plot(rya,rfPower[i,:]/1000,label=labels[i]+' (RF)',linestyle='dotted',color=colors[i])
            
            #Plot total (solid)
            ax.plot(rya,totalPower[i,:]/1000,label=labels[i]+' (Total)',color=colors[i])
            
        ax.set_xlim(0,1)
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_radial_input_power_profile.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,rfPower,nbiPower,totalPower

def integrated_power(filename,makeplot=False,saveplot=False):
    """
    This function returns the total power deposited as a function of r from 
    both RF and NBI.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    integratedNBI : np.array
        Integrated NBI power [W]. Dimensions - (ngen,rya)
    integratedRF : np.array
        Integrated RF power [W]. Dimensions - (ngen,rya)
    integratedTot : np.array
        Integrated total power [W]. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #General species labels
    labels=species_labels(filename)
    
    #Radially integrated power from NBI
    try:
        sorpw_nbii=np.array(ds['sorpw_nbii'][:])
    except:
        sorpw_nbii=np.zeros((len(rya),ngen))
    
    #Radially integrated power from RF
    try:
        sorpw_rfi=np.array(ds['sorpw_rfi'][:])
    except:
        sorpw_rfi=np.zeros((len(rya),ngen))
    
    # =========================================================================
    # Analysis
    # =========================================================================
        
    #Arrays to store the total power
    integratedNBI=[]
    integratedRF=[]
    integratedTot=[]
        
    #Go over each species
    for i in range(ngen):
        
        speciesNBI=sorpw_nbii[:,i]
        speciesRF=sorpw_rfi[:,i]
            
        speciesTot=speciesNBI+speciesRF
            
        #Add each species total to the integrated arrays
        integratedNBI.append(speciesNBI)
        integratedRF.append(speciesRF)
        integratedTot.append(speciesTot)
    
    #Convert to numpy
    integratedNBI=np.array(integratedNBI)
    integratedRF=np.array(integratedRF)
    integratedTot=np.array(integratedTot)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel(r'Integrated Power [kW]')
        ax.set_title('Integrated Power')
        
        #Go over each species
        for i in range(ngen):
            
            #Plot NBI (dashed)
            ax.plot(rya,integratedNBI[i,:]/1000,label=labels[i]+' (NBI)',linestyle='dashed',color=colors[i])
            
            #Plot RF (dotted)
            ax.plot(rya,integratedRF[i,:]/1000,label=labels[i]+' (RF)',linestyle='dotted',color=colors[i])
            
            #Plot total (solid)
            ax.plot(rya,integratedTot[i,:]/1000,label=labels[i]+' (Total)',color=colors[i])
            
        ax.set_xlim(0,1)
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_integrated_power.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
    return rya,integratedNBI,integratedRF,integratedTot

def fast_ion_confinement_time(filename,makeplot=False,saveplot=False,efastd=6):
    """
    This function calculates the fast ion confinement time as a function of the
    radius.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    efastd : float
        Boundary between warm and fast ions (keV).

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    tau_i : np.array
        Fast ion confinement time [s]. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #General species labels
    labels=species_labels(filename)
    
    #Source power density
    rya,sorpw_nbi,sorpw_rf,sorpw_tot=source_power_dens(filename)
    
    #Beam energy
    beamEnergy=25 #keV
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Convert beamEnergy to J
    beamEnergyJ=beamEnergy*1000*const.e
    
    #Beam current density
    beamCurr=sorpw_nbi/beamEnergyJ
    
    #z average of the density
    zAvgDens=[]
    #Go over each species
    for i in range(ngen):
        
        #Density
        ndwarmz,ndfz,ndtotz,solrz,solzz=ion_dens(filename,species=i,efastd=efastd)
        
        speciesAvgDens=[]
        #Go over each flux surface
        for i in range(len(solrz)):
            
            #Density array
            zDens=ndfz[i]
            
            #Take the average
            speciesAvgDens.append(np.average(zDens))
        
        zAvgDens.append(speciesAvgDens)
    
    #Convert to numpy
    zAvgDens=np.array(zAvgDens)
    
    #Fast ion confinement time
    tau_i=zAvgDens/beamCurr
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        ax.grid(True)
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel('Time [ms]')
        ax.set_title('Fast Ion Confinement Time')
        
        #Go over each species
        for i in range(ngen):
            
            ax.plot(rya,tau_i[i]*1000,label=labels[i],color=colors[i]) #Convert s to ms
            
        ax.set_xlim(0,1)
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_fast_ion_confinement_time.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
     
    return rya,tau_i

def average_energy(filename,species=0,makeplot=False,saveplot=False):
    """
    This function calculates the average energy of the ions as a function of
    the radius.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    species : int
        Index of species.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    time : np.array
        Corresponding time array [s].
    energyLastT : np.array
        Average energy of the ion species [eV]. Dimensions - (ngen,time,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)

    # =========================================================================
    # Get the raw data
    # =========================================================================

    #Energy per particle
    energym=np.array(ds['energym'][:])*1000 #Convert to eV
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Corresponding time axis
    time=np.array(ds['time'][:])
    
    #Normalized radius
    rya=np.array(ds['rya'][:])
    
    #Array with the labels for each species
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    multiSpecies=False
    
    if ngen>1:
        
        multiSpecies=True
        
        #Get energym for the right species (else it has the wrong shape)
        energym=energym[:,species,:]
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        fig,ax=plt.subplots(figsize=(12,8))

        ax.set_title('Average Particle Energy; Species='+speciesLabels[species])
        pltobj=ax.contourf(time*1000,rya,np.transpose(energym)/1000)
        ax.contour(time*1000,rya,np.transpose(energym)/1000,colors='black')
        
        ax.grid(True)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Normalized Radius (r/a)')
        ax.set_ylim(0,1)
        
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'Energy [keV]')
            
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_average_particle_energy_species_'+speciesLabels[species]+'.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,time,energym

def average_energy_final_timestep(filename,makeplot=False,saveplot=False):
    """
    This function calculates the average energy of the ions as a function of
    the radius at the final timestep.

    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    rya : np.array
        Normalized radius of the flux tube
    energyLastT : np.array
        Average energy of the ion species [eV]. Dimensions - (ngen,rya)
    """
    
    #Open the file
    ds=nc.Dataset(filename)

    # =========================================================================
    # Get the raw data
    # =========================================================================

    #Energy per particle
    energym=np.array(ds['energym'][:])*1000 #Convert to eV
    
    #Number of general (whose distribution functions are evaluated) species
    ngen=int(ds['ngen'][:])
    
    #Normalized radius
    rya=np.array(ds['rya'][:])
    
    #Array with the labels for each species
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Take the last timestep
    energyLastT=energym[-1]
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Colors for species 1 and 2
        colors=['red','blue']
        
        fig,ax=plt.subplots(figsize=(12,8))
        
        plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
        ax.grid(True)
        ax.set_title('Average Particle Energy')
        ax.set_xlabel('Normalized Radius (r/a)')
        ax.set_ylabel('Particle Energy [keV]')
        
        #Go over each species
        if ngen>=2:
            for i in range(ngen):
                ax.plot(rya,energyLastT[i]/1000,label=speciesLabels[i],color=colors[i])
        else:
            ax.plot(rya,energyLastT/1000,label=speciesLabels[0],color=colors[0])
            
        ax.set_xlim(0,1)
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_average_particle_energy_final_timestep.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return rya,energyLastT

def nbi_birth_points(filenameFreya,filenameEqdsk='',withFields=False,makeplot=False,saveplot=False):
    """
    This function returns the inital positon and velocity of the fast ions
    generated by NBI.

    Parameters
    ----------
    filenameFreya : string
        Filename for the text file with NBI information.
    filenameEqdsk : string
        Filename for the eqdsk. Only used when withFields is set to True.
    withFields : boolean
        Incorporate field lines into plots.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    xArr : np.array
        X position
    yArr : np.array
        Y position
    zArr : np.array
        Z position
    rArr : np.array
        R position
    vxArr : np.array
        X velocity
    vyArr : np.array
        Y velocity
    vzArr : np.array
        Z velocity
    """
    
    #Open the file
    fileObj=open(filenameFreya,'r')
    
    #Skip the 1st line as it does not contain data
    next(fileObj)
    
    #Initialize data arrays
    xArr=[]
    yArr=[]
    zArr=[]
    rArr=[]
    vxArr=[]
    vyArr=[]
    vzArr=[]
    
    #Go over each line in the file
    for line in csv.reader(fileObj,delimiter=' '):
        
        #Remove all 0 length elements
        lineData=list(filter(None,line))
        
        #Append the appropriate data to the lists
        xArr.append(float(lineData[1])/100)
        yArr.append(float(lineData[2])/100)
        zArr.append(float(lineData[3])/100)
        rArr.append(float(lineData[4])/100)
        vxArr.append(float(lineData[5])/100)
        vyArr.append(float(lineData[6])/100)
        vzArr.append(float(lineData[7])/100)
        
    #Convert to numpy
    xArr=np.array(xArr)
    yArr=np.array(yArr)
    zArr=np.array(zArr)
    rArr=np.array(rArr)
    vxArr=np.array(vxArr)
    vyArr=np.array(vyArr)
    vzArr=np.array(vzArr)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Initialize the plot
        fig,axs=plt.subplots(3,2,figsize=(30,3*9))
        
        fig.suptitle('Initial NBI Parameters')
        
        # =====================================================================
        # Plot initial positions
        # =====================================================================
        
        # =====================================================================
        # XZ Plot
        # =====================================================================
        
        ax=axs[0,0]
        
        if withFields==True:
            
            #Get magnetic flux surface data
            Rmesh,Zmesh,eqdsk_psi=eqTools.flux_surfaces(filenameEqdsk)
            
            #Plot levels
            PSImin=0
            PSImax=0.005710932333217801 #From genray plotting
            levels=np.arange(PSImin,PSImax,(PSImax-PSImin)/50)
            
            #Plot contours of the field
            ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
            pltObj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
        
        ax.scatter(zArr,xArr,zorder=10)
        
        ax.set_title('Initial Positions')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('X [m]')
        ax.set_xlim(-1,1)
        ax.set_ylim(-0.5,0.5)
        
        ax.grid(True)
        
        # =====================================================================
        # YZ Plot
        # =====================================================================
        
        ax=axs[1,0]
        
        if withFields==True:
            
            #Get magnetic flux surface data
            Rmesh,Zmesh,eqdsk_psi=eqTools.flux_surfaces(filenameEqdsk)
            
            #Plot levels
            PSImin=0
            PSImax=0.005710932333217801 #From genray plotting
            levels=np.arange(PSImin,PSImax,(PSImax-PSImin)/50)
            
            #Plot contours of the field
            ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
            pltObj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
        
        ax.scatter(zArr,yArr,zorder=10)
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(-1,1)
        ax.set_ylim(-0.5,0.5)
        
        ax.grid(True)
        
        # =====================================================================
        # XY Plot
        # =====================================================================
        
        ax=axs[2,0]
        
        ax.scatter(xArr,yArr)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.15,0.15)
        
        ax.grid(True)
        
        # =====================================================================
        # Plot initial velocities
        # =====================================================================
        
        # =====================================================================
        # v_z
        # =====================================================================
        
        ax=axs[0,1]
        
        ax.hist(vzArr,bins=30)
        
        ax.set_title('Initial velocities')
        ax.set_xlabel(r'$v_z$ [m/s]')
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        
        # =====================================================================
        # v_y
        # =====================================================================
        
        ax=axs[1,1]
        
        ax.hist(vyArr,bins=30)
        
        ax.set_xlabel(r'$v_y$ [m/s]')
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        
        # =====================================================================
        # v_x
        # =====================================================================
        
        ax=axs[2,1]
        
        ax.hist(vxArr,bins=30)
        
        ax.set_xlabel(r'$v_x$ [m/s]')
        ax.set_ylabel('Count')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filenameFreya.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_nbi_birth_points.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
        
    return xArr,yArr,zArr,rArr,vxArr,vyArr,vzArr

def nbi_bounce_field(filenameFreya,filenameEqdsk,makeplot=False,saveplot=False):
    """
    This function returns the magnetic field strength at which the particles
    bounce.

    Parameters
    ----------
    filenameFreya : string
        Filename for the text file with NBI information.
    filenameEqdsk : string
        Filename for the eqdsk.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Binit : np.array
        Initial field for each particle.
    Bfinal : np.array
        Bounce field for each particle.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Initial NBI parameters
    xArr,yArr,zArr,rArr,vxArr,vyArr,vzArr=nbi_birth_points(filenameFreya)
    
    #Field values
    Rmesh,Zmesh,Br,Bz,Bmag=eqTools.magnetic_field_RZ(filenameEqdsk)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    #Array to store the initial field
    Binit=[]
    
    #Go over each bounce point
    for i in range(len(xArr)):
        #Initial magnetic field
        Binit.append(field_interpolator([zArr[i],rArr[i]])[0])
    #Convert to numpy
    Binit=np.array(Binit)
    
    #Magnetic field at the bounce point
    Bfinal=Binit*(vxArr**2+vyArr**2+vzArr**2)/(vxArr**2+vyArr**2)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
    
        fig,axs=plt.subplots(1,2,figsize=(15,8))
        
        ax=axs[0]
    
        ax.set_title('Initial Magnetic Field Strength')
        ax.set_xlabel('Field Strength [T]')
        ax.set_ylabel('Count')
    
        ax.hist(Binit,bins=30)
        ax.set_xlim(0,np.max(Binit))
        
        ax.grid(True)
        
        ax=axs[1]
    
        ax.set_title('Final Magnetic Field Strength')
        ax.set_xlabel('Field Strength [T]')
    
        ax.hist(Bfinal,bins=30)
        ax.set_xlim(0,np.max(Bfinal))
        
        ax.grid(True)
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filenameFreya.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_nbi_bounce_fields.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return Binit,Bfinal

def ray_power_absorption(filename,makeplot=False,saveplot=False,species=0):
    """
    This function returns data about the higher harmonic absorption of genray
    rays.

    Parameters
    ----------
    filenameEqdsk : string
        Filename for the eqdsk.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.
    species : int
        Index of species.

    Returns
    -------
    freq : int
        Antenna frequency
    delpwr : np.array
        Power in each ray.
        It has the form- delpwr(ray number,index)
    sdpwr : np.array
        Power deposited to the ions.
        It has the form- delpwr(ray number,index)
    sbtot : np.array
        Magnetic field strength.
        It has the form- delpwr(ray number,index)
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Filename of the RF output
    #Remove the .nc part
    filenameNC=filename[0:-3]
    #Add species identifier
    filenameRF=filenameNC+'_krf00'+str(species+1)+'.nc'
    
    #Create nc object
    dsRF=nc.Dataset(filenameRF)
    
    #Number of rays
    nray=int(dsRF['nray'][:])
    
    #Antenna frequency
    freq=int(dsRF['freqcy'][:])*2*np.pi #Convert from Hz to rad/s
    
    #Power in ray channel
    delpwr=np.array(dsRF['delpwr'][:])/1e10 #Convert from erg/s to kW
    
    #Power to ions
    sdpwr=np.array(dsRF['sdpwr'][:])/1e7 #Convert from erg/s to W
    
    #Magnetic field strength
    sbtot=np.array(dsRF['sbtot'][:])/1e4 #Convert from Gauss to T
    
    #Species labels
    labels=species_labels(filename)
    
    # =========================================================================
    # Analysis
    # =========================================================================
    
    #Species Harmonics
    bArr=[]
    
    #Mass number of the species (Default --> D)
    massNum=2
    if labels[species]=='T':
        massNum=3
    
    #Get the 1st 5 harmonics
    for i in range(1,6):
        
        bHarm=freq*massNum*const.m_p/(i*const.e)
        bArr.append(bHarm)
        
    #Convert to numpy
    bArr=np.array(bArr)
    
    # =========================================================================
    # Plotting
    # =========================================================================
    
    if makeplot==True:
        
        #Initialize the plot
        fig,axs=plt.subplots(1,3,figsize=(35,10))
        
        #Title for all plots
        fig.suptitle('Higher Harmonic Absorbtion')
        
        #Evenly spaced array for changing colors
        colorArr=np.linspace(0,1,nray)
        
        # =====================================================================
        # Power in each ray channel
        # =====================================================================
        
        ax=axs[0]
        
        #Go over each ray
        for i in range(nray):
            ax.plot(delpwr[i],color=(colorArr[i],0,0))
            
        ax.set_title('Power in ray channel')
        ax.set_ylabel('Power [kW]')
        ax.set_xlabel('Index')
        
        ax.grid(True)
        
        # =====================================================================
        # Field Strength
        # =====================================================================
        
        ax=axs[1]
        
        #Go over each ray
        for i in range(nray):
            ax.plot(sbtot[i],color=(colorArr[i],0,0))
            
        ax.set_title('Field Strength')
        ax.set_ylabel('|B| [T]')
        ax.set_xlabel('Index')
        
        ax.grid(True)
        
        # =====================================================================
        # Ray power vs field strength
        # =====================================================================
        
        ax=axs[2]
        
        #Go over each ray
        for i in range(nray):
            ax.plot(sbtot[i],delpwr[i],color=(colorArr[i],0,0))
            
        #Plot cyclotron harmonics
        for i in range(len(bArr)):
            ax.plot([bArr[i],bArr[i]],[0,np.max(delpwr)],color='dodgerblue',linewidth=3)
            
        ax.set_title('Ray Power vs. Field Strength')
        ax.set_ylabel('Power [kW]')
        ax.set_xlabel('|B| [T]')
        
        ax.grid(True)
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_genray_absorbtion.png'
            
            plt.savefig(plotDest+savename,bbox_inches='tight')
        
        plt.show()
    
    return freq,delpwr,sdpwr,sbtot

def plot_dist_funcs(filename,saveplot=False,species=0,vMax=8e6):
    """
    This function plots the distribution function on all flux surfaces. If
    there are multiple general species in the run, the 'species' keyword
    selects for the one being plotted.
    
    NOTE- The plots are not normalized w.r.t. each other. This is intentional.
    
    Parameters
    ----------
    filename : string
        Location of the CQL3D output file.
    saveplot : boolean
        Save the plot.
    species : int
        Index of species.
    vMax : float
        Maximum value of the velocity in the plots [m/s].

    Returns
    -------
    None.
    """
    
    #Open the file
    ds=nc.Dataset(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Distribution function data
    f,vPar,vPerp=dist_func(filename,species=species)
    
    #Number of radial surface bins (=rdim)
    lrz=int(ds['lrz'][:])
    
    #Normalized radial mesh at bin centers
    rya=np.array(ds['rya'][:])
    
    #Array with the labels for each species
    speciesLabels=species_labels(filename)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    #Generate the savename of the plot
    #Get the name of the .nc file
    ncName=filename.split('/')[-1]
    #Remove the .nc part
    ncName=ncName[0:-3]
    savename=ncName+'_dist_func_species_'+speciesLabels[species]+'.png'
    
    #Initialize the plot
    fig,axs=plt.subplots(lrz,1,figsize=(21,lrz*9))
    
    #Go over each flux surface
    for fluxsurfplot in range(0,lrz):
    
        #Represent the r/a value in scientific notation
        ryaSciNot="{:.2e}".format(rya[fluxsurfplot])
        
        #Convert data to log
        logData=np.log10(f[fluxsurfplot])
        
        #Maximum of the distribution
        maxDist=np.round(np.max(logData))
        minDist=maxDist-15
        
        #Create the plot
        ax=axs[fluxsurfplot]
        pltobj=ax.contourf(vPar[fluxsurfplot],vPerp[fluxsurfplot],logData,levels=np.linspace(minDist,maxDist,31))
        ax.contour(pltobj,colors='black')
        ax.set_xlabel(r'$v_{||}$ [m/s]')
        ax.set_xlim(-vMax,vMax)
        ax.set_xticks(np.linspace(-vMax,vMax,17))
        ax.set_ylabel(r'$v_{\perp}$ [m/s]')
        ax.set_ylim(0,vMax)
        ax.set_title('Distribution Function (r/a = '+ryaSciNot+')')
        ax.grid(True)
        cbar=fig.colorbar(pltobj,ax=ax)
        cbar.set_label(r'log$_{10}$(v$^{-3}$)')
        
    #Save the plot
    if saveplot==True:
        plt.savefig(plotDest+savename,bbox_inches='tight')
    plt.show()