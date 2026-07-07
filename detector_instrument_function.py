# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 17:03:48 2026

@author: kunal

This code generates a predicted fusion proton detector response.

It takes as input- 
1. 2D fusion reactivity profile
2. Detector geometry and collimation
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import subprocess
import csv
import time
import random
import pickle
import numpy as np
import scipy as sc
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
from random_geometry_points.plane import Plane
from multiprocessing import Pool, cpu_count
from functools import partial

plt.rcParams.update({'font.size': 22})
plt.switch_backend('TkAgg')

def charge_to_mass_ratio(species):
    """
    This function calculates the charge to mass ratio for a given species.
    
    Available Species (Code)
    1.  Hydrogen (H)
    2.  Deuterium (D)
    3.  Tritium (T)
    4.  Helium-3 (He3)
    5.  Helium-4 (He4)
    6.  TR Hydrogen (tH)
    7.  TR Deuterium (tD)
    8.  TR Tritium (tT)
    9.  TR Helium-3 (tHe3)
    10. TR Helium-4 (tHe4)
    
    The ones labelled as time reveral (TR) have the opposite charge. This is so
    we can track their orbits backwards in time while using the same
    integrating functions.

    Parameters
    ----------
    species : string
        Species code.

    Returns
    -------
    qm : float
        Charge to mass ratio (C/kg).
    """
    
    #Initialize
    qm=0
    
    if species=='H':
        qm=const.e/const.m_p
    elif species=='D':
        qm=const.e/(const.m_n+const.m_p)
    elif species=='T':
        qm=const.e/(2*const.m_n+const.m_p)
    elif species=='He3':
        qm=2*const.e/(const.m_n+2*const.m_p)
    elif species=='He4':
        qm=2*const.e/(2*const.m_n+2*const.m_p)
    elif species=='tH':
        qm=-const.e/const.m_p
    elif species=='tD':
        qm=-const.e/(const.m_n+const.m_p)
    elif species=='tT':
        qm=-const.e/(2*const.m_n+const.m_p)
    elif species=='tHe3':
        qm=-2*const.e/(const.m_n+2*const.m_p)
    elif species=='tHe4':
        qm=-2*const.e/(2*const.m_n+2*const.m_p)
    
    return qm

def particle_mass(species):
    """
    This function calculates the mass of the particle for a given species.
    
    Available Species (Code)
    1.  Hydrogen (H)
    2.  Deuterium (D)
    3.  Tritium (T)
    4.  Helium-3 (He3)
    5.  Helium-4 (He4)
    6.  TR Hydrogen (tH)
    7.  TR Deuterium (tD)
    8.  TR Tritium (tT)
    9.  TR Helium-3 (tHe3)
    10. TR Helium-4 (tHe4)
    
    The ones labelled at time reversal (TR) are used for orbit tracking
    backwards in time. The mass does not change in this case.

    Parameters
    ----------
    species : string
        Species code.

    Returns
    -------
    m : float
        Mass of the species (in kg).
    """
    
    #Initalize
    m=0
    
    if species=='H' or species=='tH':
        m=const.m_p
    elif species=='D' or species=='tD':
        m=const.m_n+const.m_p
    elif species=='T' or species=='tT':
        m=2*const.m_n+const.m_p
    elif species=='He3' or species=='tHe3':
        m=const.m_n+2*const.m_p
    elif species=='He4' or species=='tHe4':
        m=2*const.m_n+2*const.m_p
    
    return m

def fusion_energy(species):
    """
    This function calculates the energy of the fusion product.
    
    H,T and He3 assume they are products of DD fusion.
    He4 assume it is a product of DT fusion.
    
    Available Species (Code)
    1.  Hydrogen (H)
    2.  Tritium (T)
    3.  Helium-3 (He3)
    4.  Helium-4 (He4)
    5.  TR Hydrogen (tH)
    6.  TR Tritium (tT)
    7.  TR Helium-3 (tHe3)
    8.  TR Helium-4 (tHe4)
    
    The ones labelled at time reversal (TR) are used for orbit tracking
    backwards in time. The energy does not change in this case.
    
    All values are taken from- https://en.wikipedia.org/wiki/Nuclear_fusion

    Parameters
    ----------
    species : string
        Species code.

    Returns
    -------
    energy : float
        Energy of the species (in eV).
    """
    
    #Initalize
    energy=0
    
    if species=='H' or species=='tH':
        energy=3.02e6
    elif species=='T' or species=='tT':
        energy=1.01e6
    elif species=='He3' or species=='tHe3':
        energy=0.82e6
    elif species=='He4' or species=='tHe4':
        energy=3.52e6
    
    return energy

def initialize_particle(energy,theta,phi=0,species='D'):
    """
    This function converts energy and the angles to the cartesian components
    of the velocity vector.

    Parameters
    ----------
    energy : float
        Energy of the particle (eV).
    theta : float
        Angle with the z-axis (radians).
    phi : float, optional
        Angle with the x-axis (radians).
        The default is 0.
    species : string, optional
        Species code. 
        The default is 'D'.

    Returns
    -------
    velArr : np.array
        Initial velocty of the particle in cartesian coordinates (m/s).
    """
    
    #Convert energy to Joules
    eJ=energy*const.e
    
    #Mass of the particle
    m=particle_mass(species)
        
    #Magnitude of the velocity
    velMag=np.sqrt(2*eJ/m)
    
    #Cartesian conponents of each velocity
    vx=velMag*np.sin(theta)*np.cos(phi)
    vy=velMag*np.sin(theta)*np.sin(phi)
    vz=velMag*np.cos(theta)
    
    velArr=np.array([vx,vy,vz])
    
    return velArr

def single_particle_track(xIni, energy, theta, phi=0, species='H',
                          timesteps=50, steplength=1e-9):
    """
    This function tracks the motion of the particle through the magnetic field
    as a function of the inital position and velocity.

    Uses a relativistic Boris pusher — the standard fixed-step, symplectic
    integrator for magnetic orbit tracking. It requires exactly 1 B-field
    evaluation per step (vs ~6 for RK45), with no adaptive-stepping overhead.
    Gamma is constant throughout since the Lorentz force does no work in a
    pure magnetic field.

    Parameters
    ----------
    xIni : np.array
        Initial position of the particle.
        Shape- xIni=[x0,y0,z0]
    energy : float
        Energy of the particle (eV).
    theta : float
        Angle with the z-axis (radians).
    phi : float, optional
        Angle with the x-axis (radians).
        The default is 0.
    species : string, optional
        Species code.
        The default is 'H'.
    timesteps : int
        Number of timesteps.
    steplength : float
        Size of each timestep (seconds)

    Returns
    -------
    stateVec : np.array
        State vector as a function of time (positions and velocities).
        Shape- posArr[0]=x(t)
               posArr[1]=y(t)
               posArr[2]=z(t)
               posArr[3]=vx(t)
               posArr[4]=vy(t)
               posArr[5]=vz(t)
    """

    # Get the initial velocity of the particle in cartesian coordinates
    vIni = initialize_particle(energy, theta, phi=phi, species=species)

    # Gamma is constant: Lorentz force does no work in a pure magnetic field
    speed = np.linalg.norm(vIni)
    gamma = 1.0 / np.sqrt(1.0 - (speed**2 / const.c**2))

    # Effective charge-to-mass ratio (relativistic)
    qm_gamma = charge_to_mass_ratio(species) / gamma

    # Pre-allocate output arrays
    positions  = np.empty((3, timesteps))
    velocities = np.empty((3, timesteps))

    x = xIni.copy().astype(float)
    v = vIni.copy().astype(float)

    positions[:, 0]  = x
    velocities[:, 0] = v

    dt = steplength
    half_qm_gamma_dt = 0.5 * qm_gamma * dt

    for i in range(1, timesteps):

        r = np.sqrt(x[0]**2 + x[1]**2)
        z = x[2]

        # Outside interpolation domain — particle has left the machine
        if r > 0.45 or np.abs(z) > 1.0:
            positions[:, i:]  = x[:, np.newaxis]
            velocities[:, i:] = 0.0
            break

        Br = BrInterpolator(z, r)[0]
        Bz = BzInterpolator(z, r)[0]

        if r > 0.0:
            inv_r = 1.0 / r
            Bx = Br * x[0] * inv_r
            By = Br * x[1] * inv_r
        else:
            Bx = 0.0
            By = 0.0

        # Boris rotation step (no electric field)
        tx = half_qm_gamma_dt * Bx
        ty = half_qm_gamma_dt * By
        tz = half_qm_gamma_dt * Bz
        t_dot = tx*tx + ty*ty + tz*tz
        sx = 2.0 * tx / (1.0 + t_dot)
        sy = 2.0 * ty / (1.0 + t_dot)
        sz = 2.0 * tz / (1.0 + t_dot)

        # v' = v + v × t
        vpx = v[0] + v[1]*tz - v[2]*ty
        vpy = v[1] + v[2]*tx - v[0]*tz
        vpz = v[2] + v[0]*ty - v[1]*tx

        # v_new = v + v' × s
        v[0] = v[0] + vpy*sz - vpz*sy
        v[1] = v[1] + vpz*sx - vpx*sz
        v[2] = v[2] + vpx*sy - vpy*sx

        x = x + v * dt

        positions[:, i]  = x
        velocities[:, i] = v

    return np.vstack([positions, velocities])

def compute_tracks_par(i, planePoints, energy, thetaLaunchArr, phiLaunchArr, species):
    """
    Helper function to help compute the particle tracks in parallel.
    """

    stateVec = single_particle_track(xIni = planePoints[i],
                                     energy = energy,
                                     theta = thetaLaunchArr[i],
                                     phi = phiLaunchArr[i],
                                     species = species)

    return i, stateVec

def generate_tracks_detector(detPos, detSize, detTheta, detPhi,
                             filenameEqdsk = '',
                             acceptAng = np.pi/4,
                             species = 'H',
                             numLaunchPos = 10,
                             numLaunchesPerPos = 10,
                             makeplot = False,
                             saveplot = False):
    """
    This function generates the fusion proton particle tracks for a given eqdsk
    output and a given detector position.
    
    It does this by taking 10 random points on the detector surface and
    generating 10 random initial velocity vectors for each point. These
    velocity vectors are all within the acceptance angle specified by the user.

    Parameters
    ----------
    detPos : np.array
        Position of the center of the detector.
    detSize : float
        Size of the detector in m^3. Assume circular shape.
    detTheta : float
        Angle with respect to the z axis (0 is radially outwards in cylindrical
                                          geometry)
    detPhi : float
        Angle with respect to the x axis
    filenameEqdsk : string
        Name of the eqdsk output file. Only used when plotting.
    acceptAng : float
        Acceptance angle for the detector.
        The default is 45 degrees.
    species : string, optional
        Fusion product species. 
        The default is 'H'.
        This function applies the time reversal automatically.
    numLaunchPos : int, optional
        Number of launch positions from the detector.
        The default is 10.
    numLaunchesPerPos : int, optional
        Number of particles launched per launch position.
        The default is 10.
    makeplot : boolean, optional
        Make a plot of the data.
    saveplot : boolean, optional
        Save the plot.

    Returns
    -------
    particleTracks : np.array
        Data on the state vector for each fusion proton.
    """
    
    startTime = time.time()

    # Initial energy
    energy = fusion_energy(species)
    
    # Apply time reversal to the particle
    species = 't' + species
    
    # Normal vector of the detector
    detNorm = np.array([np.sin(detTheta)*np.cos(detPhi),
                        np.sin(detTheta)*np.sin(detPhi),
                        np.cos(detTheta)])
    
    # Smallest distance of the origin to the plane
    smallDist = np.dot(detNorm,detPos)
    
    # Plane object to get random points on the plane
    plane = Plane(tuple(detNorm),
                  smallDist,
                  tuple(detPos),
                  np.sqrt(detSize/np.pi))
    
    # Get random points on the detector
    planePoints = np.asarray(plane.create_random_points(numLaunchPos))
    
    # Random theta pertubations for the particle launched
    thetaPerArr = np.random.uniform(-1, 1, numLaunchPos*numLaunchesPerPos) * acceptAng # radians
    # Random phi perturbations
    phiPerArr = np.random.uniform(-1, 1, numLaunchPos*numLaunchesPerPos) * np.sqrt(acceptAng**2-thetaPerArr**2) # radians

    # Launch theta
    thetaLaunchArr = detTheta + thetaPerArr
    # Launch phi
    phiLaunchArr = detPhi + phiPerArr
    
    # Get the particle tracks
    particleTracks = []
    
    # Particle number
    particleNum = 1
    
    setupTime = time.time()
    # print('Time taken to set up track generation- '+str(setupTime-startTime)+' seconds')

    # Go over each point
    for point in planePoints:
        
        # Random perturbations per point
        for i in range(numLaunchesPerPos):
        
            # Random theta perturbation
            thetaPer = random.uniform(-1,1) * acceptAng # [radians]
            # Random phi perturbation
            phiPer = random.uniform(-1,1) * np.sqrt(acceptAng**2 - thetaPer**2)
            
            # Launch theta
            thetaLaunch = detTheta + thetaPer
            # Launch phi
            phiLaunch = detPhi + phiPer
            
            # Increment
            particleNum += 1
            
            # Single particle track
            stateVec = single_particle_track(xIni = point,
                                             energy = energy,
                                             theta = thetaLaunch,
                                             phi = phiLaunch,
                                             species = species)
            
            particleTracks.append(stateVec)
            
    # Convert to numpy
    particleTracks = np.array(particleTracks)
    
    if makeplot==True:
        
        # eqdsk mesh arrays
        if filenameEqdsk != '':
            Rmesh,Zmesh,eqdsk_psi=eqTools.flux_surfaces(filename=filenameEqdsk)
        else:
            print('Cannot make plot, eqdsk filename not provided')
            return particleTracks
        
        #Create the plot
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        
        #Plot the poloidal flux
        #Plot levels
        levels=np.linspace(0,psilim,40)
        
        pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
        pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')
        
        #Plot each ray
        for i in range(len(particleTracks)):
            
            ax.plot(particleTracks[i][2],particleTracks[i][0],color='red')
            
            #Plot the end points
            #Only add label to 1 track
            if i==1:
                ax.scatter(particleTracks[i][2][-1],particleTracks[i][0][-1],zorder=5,color='blue',label='End Point')
            else:
                ax.scatter(particleTracks[i][2][-1],particleTracks[i][0][-1],zorder=5,color='blue')
            #Plot the start points
            #Only add label to 1 track
            if i==1:
                ax.scatter(particleTracks[i][2][0],particleTracks[i][0][0],s=75,zorder=5,color='black',label='Start Point')
            else:
                ax.scatter(particleTracks[i][2][0],particleTracks[i][0][0],s=75,zorder=5,color='black')
                
        #Plot the detector
        if type(detPos)!=int:
            
            #Detector normal
            ax.quiver(detPos[2],detPos[0],detNorm[2],detNorm[0],width=3e-3,zorder=10)
            # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
            
            #Detector boundary
            #100 points on a unit circle in the xy-plane
            detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
            #Scale unit circle to detector size
            detPosArr*=np.sqrt(detSize/np.pi)
            #Initialize the rotation
            rotObj=R.from_euler('ZYX',[detPhi,detTheta,0])
            #Apply the rotation to each point
            rotArr=[]
            for i in range(len(detPosArr[0])):
                rotArr.append(rotObj.apply(detPosArr[:,i]))
            rotArr=np.array(rotArr)
            #Move to the detector location
            rotArr+=detPos
            #Plot the detector boundary
            ax.plot(rotArr[:,2],rotArr[:,0],label='Detector',linewidth=6,zorder=10)
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('X [m]')
        
        ax.set_xlim(np.min(Zmesh),np.max(Zmesh))
        ax.set_ylim(-np.max(Rmesh),np.max(Rmesh))
        
        ax.legend()
        ax.grid(True)
        
        if saveplot==True:
                
                #Generate the savename of the plot
                #Get the name of the .nc file
                ncName=filenameEqdsk.split('/')[-1]
                #Remove the .nc part
                ncName=ncName[0:-3]
                savename=ncName+'_launch_from_detector.png'
                
                plt.savefig(plotDest+savename,bbox_inches='tight')
        
        plt.show()
    
    endTime = time.time()
    # print('Time taken to generate tracks in generate_tracks_detector '+str(endTime-startTime)+' seconds')

    return particleTracks

def generate_tracks_aperture(detPos, detPhi, detSize, bendRad, tubeAng,
                             species='H', numLaunchPos=20, numLaunchesPerPos=20,
                             makeplot=False, saveplot=False, savename=None):
    """
    This function calculates the particle tracks that hit the detector and pass
    through the given aperture geometry.
    
    Essentially, this function is the same as generate_tracks_detector but with
    the added filtering of removing anything that doesn't pass through the
    specified aperture.
    
    The aperture geometry that is specified here is of the 'bent pipe' variety
    as that is the simplest to model and manufacture and gives good spatial
    localization for the detector.

    Parameters
    ----------
    detPos : np.array
        Position of the center of the detector.
        [x, y, z]
    detPhi : float
        Angle the detector makes with the x-axis in the XY plane. [radians]
    detSize : float
        Size of the detector (assume circular shape). [m^2]
    bendRad : float
        Bend radius of the collimating tube. [m]
    tubeAng : float
        Angle subtended by the collimating tube. [radians]
    species : string
        Name of the fusion product species.
        Default = 'H'
        This function applies the time reversal automatically.
    numLaunchPos : int
        Number of launch positions from the detector surface.
        Default = 20
    numLaunchPerPos : int
        Number of particles launch per launch position.
        Default = 20
    makeplot : boolean, optional
        Make a plot of the data.
    saveplot : boolean, optional
        Save the plot.
    savename : string, optional
        Name of the .pkl file to save the (x,y) of the particle tracks that pass through the aperture.
        Default = None (does not save).

    Returns
    -------
    openingTracks : list
        All the state vectors for the fusion products that hit the detector and
        make it through the aperture.
        1st index- Track number
    """
    
    # =============================================================================
    # Calculate the position/orientation of the tube opening
    # =============================================================================

    # Theta (angle wrt z axis)  ***DO NOT CHANGE***
    detTheta = (np.pi/180) * 90 # radians

    # Length of chord from tube opening to detector (law of cosines)
    chordLen = bendRad * np.sqrt(2*(1-np.cos(tubeAng)))

    # Place the detector 1 chord length from the origin
    localPos = np.array([chordLen,0,0])

    # Net angle by which we need to rotate the opening positon
    netAng = detPhi + tubeAng

    # Rotation matrix (-ve angle since we are rotating the point and not the axes)
    rotMatrix = np.array([[np.cos(-netAng),np.sin(-netAng)],
                          [-np.sin(-netAng),np.cos(-netAng)]])

    # Rotate the opening position
    localPos[:2] = np.matmul(rotMatrix,localPos[:2])

    # Go from local to global coordinates
    adjustedPos = localPos + detPos

    # Normal vector of the detector
    detNorm = np.array([np.sin(detTheta)*np.cos(detPhi),
                        np.sin(detTheta)*np.sin(detPhi),
                        np.cos(detTheta)])

    # Normal vector of the opening
    openNorm = np.array([np.sin(detTheta)*np.cos(netAng),
                         np.sin(detTheta)*np.sin(netAng),
                         np.cos(detTheta)])

    # =============================================================================
    # Generate the particle tracks
    # =============================================================================

    # Particle species
    species='H'

    # Get the particle tracks
    startTime = time.time()

    particleTracks=generate_tracks_detector(detPos, detSize, detTheta, detPhi,
                                            species=species, acceptAng=(np.pi/180)*15,
                                            numLaunchPos=numLaunchPos, numLaunchesPerPos=numLaunchesPerPos)
    
    generationTime = time.time()
    # print('Time taken to generate tracks- '+str(generationTime-startTime)+' seconds')

    # =============================================================================
    # Analysis
    # =============================================================================

    #Array to store the particle tracks until they leave the core
    coreTracks=[]

    #Radius of the core
    coreRad=0.1 #meters

    #Go over each particle track
    for i in range(len(particleTracks)):
        
        # print('Track number- '+str(i+1))
        
        #Get the current track
        currTrack=particleTracks[i]

        #Check if the particle when through the core
        throughTheCore,coreTrack=through_the_core(currTrack,coreRad)
        
        if throughTheCore==True:
            coreTracks.append(coreTrack)
                
    #Array to store the tracks that go through the opening
    openingTracks=[]

    #Go over each particle track to see if it goes through the opening
    for i in range(len(coreTracks)):
        
        # print('Track number- '+str(i+1))
        
        # Get the current track
        currTrack = coreTracks[i]
        
        # Check if the particle went through the opening
        detectorHit, hitPos, hitVel, hitTrack = hit_detector(currTrack, adjustedPos, openNorm, detSize)
        
        if detectorHit == True:
            openingTracks.append(currTrack)
            
    # Save the (x,z) of the tracks that go through the opening
    if savename != None:

        # Get the (x,z) of the tracks that go through the opening
        xzTracks = []
        for track in openingTracks:
            xzTracks.append([track[0], track[1]])
        
        # Save the (x,z) of the tracks that go through the opening
        with open(savename, 'wb') as f:
            pickle.dump(xzTracks, f)

    # =============================================================================
    # Plotting
    # =============================================================================
    
    if makeplot == True:
        
        plt.rcParams.update({'font.size': 32})

        #Relative sizes of each plot
        relSize=dict(width_ratios=[1.5,1],height_ratios=[1,1])

        detectorLoc=detPos
        detectorVec=detNorm
        detectorArea=detSize

        fig,axs=plt.subplot_mosaic([['upper left','right'],['lower left','right']],gridspec_kw=relSize,figsize=(28,11),constrained_layout=True)

        #Go over each subplot
        for label,ax in axs.items():
            
            #XZ plot
            if label=='lower left':
                 
                #Plot each ray
                if len(openingTracks)!=0:
                    for i in range(len(openingTracks)):
                        
                        ax.plot(openingTracks[i][2],openingTracks[i][0],color='red')
                    
                        #Plot the endpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][2][-1],openingTracks[i][0][-1],zorder=5,color='blue',label='End Point')
                        else:
                            ax.scatter(openingTracks[i][2][-1],openingTracks[i][0][-1],zorder=5,color='blue')
                        #Plot the startpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][2][0],openingTracks[i][0][0],s=75,zorder=5,color='black',label='Start Point')
                        else:
                            ax.scatter(openingTracks[i][2][0],openingTracks[i][0][0],s=75,zorder=5,color='black')
                
                #Plot the poloidal flux
                #Plot levels
                levels=np.linspace(0,psilim,40)
                
                pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
                pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
                
                #Plot the detector
                if type(detectorLoc)!=int:
                    
                    #Detector normal
                    ax.quiver(detectorLoc[2],detectorLoc[0],detectorVec[2],detectorVec[0],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(detectorVec[2])
                    phi=np.arctan2(detectorVec[1],detectorVec[0])
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=detectorLoc
                    #Plot the detector boundary
                    ax.plot(rotArr[:,2],rotArr[:,0],label='Detector',linewidth=6,zorder=10)
                    
                #Plot the tube opening
                if type(adjustedPos)!=int:
                    
                    #Detector normal
                    ax.quiver(adjustedPos[2],adjustedPos[0],openNorm[2],openNorm[0],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(openNorm[2])
                    phi=np.arctan2(openNorm[1],openNorm[0])
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=adjustedPos
                    #Plot the detector boundary
                    ax.plot(rotArr[:,2],rotArr[:,0],label='Opening',linewidth=6,zorder=10)
                
                #Axes labels
                ax.set_xlabel('Z [m]')
                ax.set_ylabel('X [m]')
                
                #Subplot title
                ax.set_title('XZ Plane')
                
                # #Zoom into the area with particle tracks
                # if len(openingTracks)!=0:
                #     #x points of each track
                #     xVals=[openingTracks[i][0][:] for i in range(0,len(openingTracks))]                
                #     #Pad array with nan values
                #     xVals=pad_array(xVals)
                #     xMax=np.nanmax(xVals)
                #     xMin=np.nanmin(xVals)
                #     #z points of each track
                #     zVals=[openingTracks[i][2][:] for i in range(0,len(openingTracks))]
                #     #Pad array with nan values
                #     zVals=pad_array(zVals)
                #     zMax=np.nanmax(zVals)
                #     zMin=np.nanmin(zVals)
                    
                #     ax.set_xlim(zMin-0.25,zMax+0.25)
                #     ax.set_ylim(xMin-0.05,xMax+0.05)
                
                ax.set_ylim(-0.3,0.3)
                ax.set_xlim(-0.8,0.8)
                
                # ax.legend()
                ax.grid(True)

            #YZ plot
            if label=='upper left':
                
                #Plot each ray
                if len(openingTracks)!=0:
                    for i in range(len(openingTracks)):
                        
                        ax.plot(openingTracks[i][2],openingTracks[i][1],color='red')
                    
                        #Plot the endpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][2][-1],openingTracks[i][1][-1],zorder=5,color='blue',label='End Point')
                        else:
                            ax.scatter(openingTracks[i][2][-1],openingTracks[i][1][-1],zorder=5,color='blue')
                        #Plot the startpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][2][0],openingTracks[i][1][0],s=75,zorder=5,color='black',label='Start Point')
                        else:
                            ax.scatter(openingTracks[i][2][0],openingTracks[i][1][0],s=75,zorder=5,color='black')
                
                #Plot the poloidal flux
                #Plot levels
                levels=np.linspace(0,psilim,40)
                
                pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
                pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
                
                #Plot the detector
                if type(detectorLoc)!=int:
                    
                    #Detector normal
                    ax.quiver(detectorLoc[2],detectorLoc[1],detectorVec[2],detectorVec[1],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(detectorVec[2])
                    phi=np.arctan2(detectorVec[1],detectorVec[0])
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=detectorLoc
                    #Plot the detector boundary
                    ax.plot(rotArr[:,2],rotArr[:,1],label='Detector',linewidth=6,zorder=10)
                    
                #Plot the tube opening
                if type(adjustedPos)!=int:
                    
                    #Detector normal
                    ax.quiver(adjustedPos[2],adjustedPos[1],openNorm[2],openNorm[1],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(openNorm[2])
                    phi=np.arctan2(openNorm[1],openNorm[0])            
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=adjustedPos
                    #Plot the detector boundary
                    ax.plot(rotArr[:,2],rotArr[:,1],label='Opening',linewidth=6,zorder=10)
                
                ax.set_xlabel('Z [m]')
                ax.set_ylabel('Y [m]')
                
                #Subplot title
                ax.set_title('YZ Plane')
                
                # #Zoom into the area with particle tracks
                # if len(openingTracks)!=0:
                #     #x points of each track
                #     xVals=[openingTracks[i][1][:] for i in range(0,len(openingTracks))]                
                #     #Pad array with nan values
                #     xVals=pad_array(xVals)
                #     xMax=np.nanmax(xVals)
                #     xMin=np.nanmin(xVals)
                #     #z points of each track
                #     zVals=[openingTracks[i][2][:] for i in range(0,len(openingTracks))]
                #     #Pad array with nan values
                #     zVals=pad_array(zVals)
                #     zMax=np.nanmax(zVals)
                #     zMin=np.nanmin(zVals)
                    
                #     ax.set_xlim(zMin-0.25,zMax+0.25)
                #     ax.set_ylim(xMin-0.05,xMax+0.05)
                
                ax.set_ylim(-0.3,0.3)
                ax.set_xlim(-0.8,0.8)
                
                # ax.legend()
                ax.grid(True)

            #XY plot
            if label=='right':
                
                #Plot each ray
                if len(openingTracks)!=0:
                    for i in range(len(openingTracks)):
                        
                        ax.plot(openingTracks[i][0],openingTracks[i][1],color='red')
                    
                        #Plot the endpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][0][-1],openingTracks[i][1][-1],zorder=5,color='blue',label='End Point')
                        else:
                            ax.scatter(openingTracks[i][0][-1],openingTracks[i][1][-1],zorder=5,color='blue')
                        #Plot the startpoints
                        #Only add label to 1 track
                        if i==1:
                            ax.scatter(openingTracks[i][0][0],openingTracks[i][1][0],s=75,zorder=5,color='black',label='Start Point')
                        else:
                            ax.scatter(openingTracks[i][0][0],openingTracks[i][1][0],s=75,zorder=5,color='black')
                
                #Plot the r=10cm circle
                #Create the circle object
                circ=plt.Circle((0,0),  #Center
                          0.1,          #Radius in m
                          color='g',    #Color
                          linewidth=5,  #Line width
                          fill=False)   #Only make an outline
                #Add to the plot
                ax.add_patch(circ)
                
                #Plot the detector
                if type(detectorLoc)!=int:
                    
                    #Detector normal
                    ax.quiver(detectorLoc[0],detectorLoc[1],detectorVec[0],detectorVec[1],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(detectorVec[2])
                    phi=np.arctan2(detectorVec[1],detectorVec[0])
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=detectorLoc
                    #Plot the detector boundary
                    ax.plot(rotArr[:,0],rotArr[:,1],label='Detector',linewidth=6,zorder=10)
                    
                #Plot the tube opening
                if type(adjustedPos)!=int:
                    
                    #Detector normal
                    ax.quiver(adjustedPos[0],adjustedPos[1],openNorm[0],openNorm[1],width=3e-3,zorder=10)
                    # ax.quiverkey(detNormalObj,0.9,0.9,1,label='Detector Normal',labelpos='E')
                    
                    #Detector boundary
                    #100 points on a unit circle in the xy-plane
                    detPosArr=np.array([np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),np.zeros(100)])
                    #Scale unit circle to detector size
                    detPosArr*=np.sqrt(detectorArea/np.pi)
                    #Angles by which to rotate this circle based on the detector normal
                    theta=np.arccos(openNorm[2])
                    phi=np.arctan2(openNorm[1],openNorm[0])
                    #Initialize the rotation
                    rotObj=R.from_euler('ZYX',[phi,theta,0])
                    #Apply the rotation to each point
                    rotArr=[]
                    for i in range(len(detPosArr[0])):
                        rotArr.append(rotObj.apply(detPosArr[:,i]))
                    rotArr=np.array(rotArr)
                    #Move to the detector location
                    rotArr+=adjustedPos
                    #Plot the detector boundary
                    ax.plot(rotArr[:,0],rotArr[:,1],label='Opening',linewidth=6,zorder=10)
                
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                
                #Subplot title
                ax.set_title('XY Plane')
                
                #Zoom in to make plots clearer
                ax.set_xlim(-0.4,0.4)
                ax.set_ylim(-0.4,0.4)
                
                ax.legend()
                ax.grid(True)
                
            #Plot title
            fig.suptitle(r'$\phi$='+str(int(detPhi*180/np.pi))+r'$^{\circ}$'+
                         r'; Bend Radius= '+str(bendRad)+'m; Tube Section= '+str(int(tubeAng*180/np.pi))+r'$^{\circ}$')

        # =============================================================================
        # Save the plot
        # =============================================================================
        if saveplot == True:

            # Get the name of the eqdsk file
            ncName = filenameEqdsk.split('/')[-1]
    
            # Generate the savename
            savename = ncName+'_proton_detector_r_'+str(int(detPos[0]*100))+'cm_z_'+str(int(detPos[2]*100))+'cm_theta_'+str(int(detTheta*180/np.pi))+'deg_phi_'+str(int(detPhi*180/np.pi))+'deg_with_collimator.png'
    
            print(savename)
            plt.savefig('/home/sanwalka/synthetic_proton_detector/plots/' + savename + '.png', 
                        bbox_inches='tight')

            plt.close()
        
        elif saveplot == False:
            
            plt.show()
    
    endTime = time.time()
    # print('Time taken to analyze and plot tracks- '+str(endTime-generationTime)+' seconds')

    return openingTracks

def volume_weights(detPos, detPhi, detSize, bendRad, tubeAng, 
                   cellSize=1e-2, errorLim=1e-2, minParticles=50, maxParticles=200,
                   makeplot=False, savename=None):
    """
    This function calculates the instrument function for a given detector 
    geometry. It also calculates the distance between the detector aperture and 
    each voxel.
    
    It will run until the either the error in weights goes below OR
    the maximum number of launches particles is reached, whichever comes first.
    This is done as this function can often take >2min to run and so prevents
    the code for running too long.

    Parameters
    ----------
    detPos : np.array
        Position of the center of the detector.
        [x, y, z]
    detPhi : float
        Angle the detector makes with the x-axis in the XY plane. [radians]
    detSize : float
        Size of the detector (assume circular shape). [m^2]
    bendRad : float
        Bend radius of the collimating tube. [m]
    tubeAng : float
        Angle subtended by the collimating tube. [radians]
    cellSize : float, optional
        Size (in meters) of the mesh cell. The default is 1e-2.
    errorLim : float, optional
        Error limit before the simulation is considered to have converged.
        The default is 1e-2.
        If set to None, then the simulation will only stop when maxParticles is reached.
    minParticles : float, optional
        Minimum number of particles to be launched by the simulation before it
        exits.
        The default is 50.
    maxParticles : float, optional
        Maximum number of particles to be launched by the simulation before it
        exits.
        The default is 200.
    makeplot : boolean, optional
        Make a plot of the data. Default is False.
    savename : string, optional
        Save the data that is being plotted. Default is None.

    Returns
    -------
    threeDPoints : np.array
        Coordinates of the mesh points.
        '1D' array where each element is the 3 coordinates of that mesh point.
    volumeWeights : np.array
        Corresponding detector weight sensitivity.
    volumeDistances : np.array
        Distances from the aperture/collimation opening to the voxel.
        If no particle hits the detector, this is set to 0.
    errorArr : list
        Error rate evolution.
    totParticles : list
        Number of particles tracked evolution.
    """
    
    xArr = np.arange(-0.1, 0.1+cellSize, cellSize)
    yArr = xArr
    zArr = np.arange(-1, 1+2e-2, 2e-2)

    # Create an array of the 3D points
    # 1st index- point number
    # 2nd index- [x,y,z]
    threeDPoints = np.vstack(np.meshgrid(xArr, yArr ,zArr)).reshape(3,-1).T

    # Array to store the weights of each 3D point
    volumeWeights = np.zeros(shape=(len(threeDPoints)))
    
    # Array to store the distance of each voxel from the detector
    volumeDistances = np.zeros(shape=(len(threeDPoints)))

    # Current error
    currError = 1
    # Track the error
    errorArr = [1]
    # Track the number of particles
    totParticles = [0]

    # Launch particles until the error rate is below errorLim or the maximum
    # number of particles is reached
    i = 0 # Track number of iterations through the loop

    # If errorLim is None, then we only stop when maxParticles is reached
    if errorLim != None:

        while (currError >= errorLim and totParticles[-1] <= maxParticles) or totParticles[-1] <= minParticles:
        
            startTime = time.time()
            
            # Old volume weights to track error
            oldVolumeWeights = np.copy(volumeWeights)
            
            # Generate particle tracks given the detector geometry
            openingTracks = generate_tracks_aperture(detPos=detPos, 
                                                     detPhi=detPhi, 
                                                     detSize=detSize, 
                                                     bendRad=bendRad, 
                                                     tubeAng=tubeAng,
                                                     numLaunchPos=5,
                                                     numLaunchesPerPos=5,
                                                     makeplot=False)
            
            tracksGenerationTime = time.time()
            # print(f'Time taken to generate tracks= {tracksGenerationTime - startTime}s')

            # Go over each particle track and see which volume elements it went through
            chunk_size = 10000  # Process grid points in chunks to avoid huge memory usage
            
            for j in range(len(openingTracks)):
                
                currTrack = openingTracks[j][:3]  # Only keep the position data (3, num_positions)
                all_positions = currTrack.T  # shape: (num_positions, 3)
                num_positions = all_positions.shape[0]
                
                # Calculate cumulative distance along track for each position
                # distances[i] = total distance from detector (position 0) to position i
                if num_positions > 1:
                    # Distance between consecutive positions
                    deltas = np.diff(all_positions, axis=0)  # (num_positions-1, 3)
                    segment_lengths = np.linalg.norm(deltas, axis=1)  # (num_positions-1,)
                    # Cumulative distance from start
                    cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])  # (num_positions,)
                else:
                    cumulative_dist = np.array([0])
                
                # Process grid points in chunks
                num_grid_points = len(threeDPoints)
                
                for chunk_start in range(0, num_grid_points, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_grid_points)
                    grid_chunk = threeDPoints[chunk_start:chunk_end]  # (chunk_size, 3)
                    
                    # Reshape for broadcasting:
                    # grid_chunk: (chunk_size, 3) -> (chunk_size, 1, 3)
                    # all_positions: (num_positions, 3) -> (1, num_positions, 3)
                    grid_reshaped = grid_chunk[:, np.newaxis, :]
                    positions_reshaped = all_positions[np.newaxis, :, :]
                    
                    # Calculate distances for this chunk
                    # Result shape: (chunk_size, num_positions, 3)
                    distances = np.abs(grid_reshaped - positions_reshaped)
                    
                    # Find which grid points are within half cell size for each position
                    # Shape: (chunk_size, num_positions)
                    within_range = np.all(distances <= 0.5 * cellSize, axis=2)
                    
                    # For each grid point, find if it was hit and at which position index
                    hit_mask = np.any(within_range, axis=1)  # (chunk_size,) - which points were hit
                    
                    # For hit points, find the FIRST position that hit them
                    first_hit_indices = np.full(len(grid_chunk), -1, dtype=int)
                    first_hit_indices[hit_mask] = np.argmax(within_range[hit_mask], axis=1)
                    
                    # Update volumeWeights and volumeDistances
                    global_indices = np.arange(chunk_start, chunk_end)
                    hit_global_indices = global_indices[hit_mask]
                    hit_position_indices = first_hit_indices[hit_mask]
                    
                    # Increment weights
                    volumeWeights[hit_global_indices] += 1
                    
                    # Update distances (use the cumulative distance at the hit position)
                    volumeDistances[hit_global_indices] = cumulative_dist[hit_position_indices]
                        
            # Update the error
            if len(openingTracks) > 0:
                
                currError = np.sum(volumeWeights - oldVolumeWeights)/np.sum(volumeWeights)
                
                errorArr.append(currError)
                totParticles.append(totParticles[-1] + len(openingTracks))
                
                # print('Error = {}'.format(currError))
                # print('Particle Number = {}'.format(totParticles[-1]))
                
            volumeTrackingTime = time.time()
            # print(f'Time taken to check volumes = {volumeTrackingTime - tracksGenerationTime}s')
                
            i+=1
    
    else:

        while totParticles[-1] <= maxParticles:
        
            startTime = time.time()
            
            # Old volume weights to track error
            oldVolumeWeights = np.copy(volumeWeights)
            
            # Generate particle tracks given the detector geometry
            openingTracks = generate_tracks_aperture(detPos=detPos, 
                                                    detPhi=detPhi, 
                                                    detSize=detSize, 
                                                    bendRad=bendRad, 
                                                    tubeAng=tubeAng,
                                                    numLaunchPos=5,
                                                    numLaunchesPerPos=5,
                                                    makeplot=False)
            
            tracksGenerationTime = time.time()
            # print(f'Time taken to generate tracks= {tracksGenerationTime - startTime}s')

            # Go over each particle track and see which volume elements it went through
            chunk_size = 10000  # Process grid points in chunks to avoid huge memory usage
            
            for j in range(len(openingTracks)):
                
                currTrack = openingTracks[j][:3]  # Only keep the position data (3, num_positions)
                all_positions = currTrack.T  # shape: (num_positions, 3)
                num_positions = all_positions.shape[0]
                
                # Calculate cumulative distance along track for each position
                # distances[i] = total distance from detector (position 0) to position i
                if num_positions > 1:
                    # Distance between consecutive positions
                    deltas = np.diff(all_positions, axis=0)  # (num_positions-1, 3)
                    segment_lengths = np.linalg.norm(deltas, axis=1)  # (num_positions-1,)
                    # Cumulative distance from start
                    cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])  # (num_positions,)
                else:
                    cumulative_dist = np.array([0])
                
                # Process grid points in chunks
                num_grid_points = len(threeDPoints)
                
                for chunk_start in range(0, num_grid_points, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_grid_points)
                    grid_chunk = threeDPoints[chunk_start:chunk_end]  # (chunk_size, 3)
                    
                    # Reshape for broadcasting:
                    # grid_chunk: (chunk_size, 3) -> (chunk_size, 1, 3)
                    # all_positions: (num_positions, 3) -> (1, num_positions, 3)
                    grid_reshaped = grid_chunk[:, np.newaxis, :]
                    positions_reshaped = all_positions[np.newaxis, :, :]
                    
                    # Calculate distances for this chunk
                    # Result shape: (chunk_size, num_positions, 3)
                    distances = np.abs(grid_reshaped - positions_reshaped)
                    
                    # Find which grid points are within half cell size for each position
                    # Shape: (chunk_size, num_positions)
                    within_range = np.all(distances <= 0.5 * cellSize, axis=2)
                    
                    # For each grid point, find if it was hit and at which position index
                    hit_mask = np.any(within_range, axis=1)  # (chunk_size,) - which points were hit
                    
                    # For hit points, find the FIRST position that hit them
                    first_hit_indices = np.full(len(grid_chunk), -1, dtype=int)
                    first_hit_indices[hit_mask] = np.argmax(within_range[hit_mask], axis=1)
                    
                    # Update volumeWeights and volumeDistances
                    global_indices = np.arange(chunk_start, chunk_end)
                    hit_global_indices = global_indices[hit_mask]
                    hit_position_indices = first_hit_indices[hit_mask]
                    
                    # Increment weights
                    volumeWeights[hit_global_indices] += 1
                    
                    # Update distances (use the cumulative distance at the hit position)
                    volumeDistances[hit_global_indices] = cumulative_dist[hit_position_indices]
                        
            # Update the error
            if len(openingTracks) > 0:
                
                currError = np.sum(volumeWeights - oldVolumeWeights)/np.sum(volumeWeights)
                
                errorArr.append(currError)
                totParticles.append(totParticles[-1] + len(openingTracks))
                
                # print('Error = {}'.format(currError))
                # print('Particle Number = {}'.format(totParticles[-1]))
                
            volumeTrackingTime = time.time()
            # print(f'Time taken to check volumes = {volumeTrackingTime - tracksGenerationTime}s')
                
            i+=1

    print(f'Number of loop iterations = {i}')
    print(f'Final error = {currError}')
    print(f'Total number of particles launched = {totParticles[-1]}')
        
    # Normalize the volume weights
    volumeWeights /= np.max(volumeWeights)
    
    if makeplot == True:
        
        # =============================================================================
        # Calculate the position/orientation of the tube opening
        # =============================================================================

        # Theta (angle wrt z axis)  ***DO NOT CHANGE***
        detTheta = (np.pi/180) * 90 # radians

        # Length of chord from tube opening to detector (law of cosines)
        chordLen = bendRad * np.sqrt(2*(1-np.cos(tubeAng)))

        # Place the detector 1 chord length from the origin
        localPos = np.array([chordLen,0,0])

        # Net angle by which we need to rotate the opening positon
        netAng = detPhi + tubeAng

        # Rotation matrix (-ve angle since we are rotating the point and not the axes)
        rotMatrix = np.array([[np.cos(-netAng),np.sin(-netAng)],
                                [-np.sin(-netAng),np.cos(-netAng)]])

        # Rotate the opening position
        localPos[:2] = np.matmul(rotMatrix,localPos[:2])

        # Go from local to global coordinates
        adjustedPos = localPos + detPos

        # Normal vector of the detector
        detNorm = np.array([np.sin(detTheta)*np.cos(detPhi),
                            np.sin(detTheta)*np.sin(detPhi),
                            np.cos(detTheta)])

        # Normal vector of the opening
        openNorm = np.array([np.sin(detTheta)*np.cos(netAng),
                                np.sin(detTheta)*np.sin(netAng),
                                np.cos(detTheta)])

        detectorVec = detNorm
        detectorLoc = detPos

        # Indices of threeDPoints that are closest to the detector
        zPoints = threeDPoints[:, 2]
        idx = np.abs(zPoints-detPos[2]) < cellSize
        
        # x and y positions
        xPosPlotting = threeDPoints[idx, 0]
        yPosPlotting = threeDPoints[idx, 1]
        
        #### Plot the volume weights
        
        fig = plt.figure(figsize=(12, 10), tight_layout='True')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        
        # Get the weights at the closest z-position of the detector
        plottingWeights = volumeWeights[idx]
        
        # Plot the weights
        pltObj1 = ax.tricontourf(xPosPlotting, yPosPlotting, plottingWeights,
                                 levels = np.linspace(0, 1, 100),
                                 cmap='inferno')
        
        # Detector boundary
        # 100 points on a unit circle in the xy-plane
        detPosArr = np.array([np.cos(np.linspace(0,2*np.pi,100)),
                              np.sin(np.linspace(0,2*np.pi,100)),
                              np.zeros(100)])
        # Scale unit circle to detector size
        detPosArr *= np.sqrt(detSize/np.pi)
        # Angles by which to rotate this circle based on the detector normal
        theta = np.arccos(detectorVec[2])
        phi = np.arctan2(detectorVec[1],detectorVec[0])
        # Initialize the rotation
        rotObj = R.from_euler('ZYX',[phi,theta,0])
        # Apply the rotation to each point
        rotArr = []
        for i in range(len(detPosArr[0])):
            rotArr.append(rotObj.apply(detPosArr[:,i]))
        rotArr = np.array(rotArr)
        # Move to the detector location
        rotArr += detectorLoc
        # Plot the detector boundary
        ax.plot(rotArr[:,0],rotArr[:,1],
                label='Detector',
                linewidth=6,
                zorder=10)
        
        # Opening boundary
        # Angles by which to rotate this circle based on the detector normal
        theta = np.arccos(openNorm[2])
        phi = np.arctan2(openNorm[1],openNorm[0])
        # Initialize the rotation
        rotObj = R.from_euler('ZYX',[phi,theta,0])
        # Apply the rotation to each point
        rotArr2 = []
        for i in range(len(detPosArr[0])):
            rotArr2.append(rotObj.apply(detPosArr[:,i]))
        rotArr2 = np.array(rotArr2)
        # Move to the detector location
        rotArr2 += adjustedPos
        # Plot the opening boundary
        ax.plot(rotArr2[:,0], rotArr2[:,1],
                label='Opening',
                linewidth=6,
                zorder=10)
        
        ax.legend()
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Normalized Response Function (z={np.round(detPos[2], 2)}m)')
        
        fig.colorbar(pltObj1, ticks=np.linspace(0, 1, 6))
        
        plt.show()

        if savename != None:

            # Save the data being plotted
            np.savez('/home/sanwalka/synthetic_proton_detector/reactivity/'+savename+'.npz',
                     detPosX = rotArr[:, 0],
                     detPosY = rotArr[:, 1],

                     openingPosX = rotArr2[:, 0],
                     openingPosY = rotArr2[:, 1],

                     xPosWeights = xPosPlotting,
                     yPosWeights = yPosPlotting,
                     weights = volumeWeights[idx],
                     distance = volumeDistances[idx],
                     
                     detZLoc = detPos[2])
        
        #### Plot the distances to the detector
        
        fig = plt.figure(figsize=(12, 10), tight_layout='True')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        
        # Get the weights at the closest z-position of the detector
        plottingDist = volumeDistances[idx]
        
        # Plot the weights
        pltObj2 = ax.tricontourf(xPosPlotting, yPosPlotting, plottingDist,
                                 levels = np.linspace(0, np.max(plottingDist), 100),
                                 cmap='inferno')
        
        # Plot the detector
        ax.scatter([detPos[0]], [detPos[1]], s=200,
                   label='Detector')
        
        # Length of chord from tube opening to detector (law of cosines)
        chordLen = bendRad * np.sqrt(2*(1-np.cos(tubeAng)))
        # Plot the aperture
        ax.scatter([detPos[0] + np.cos(detPhi+tubeAng)*chordLen], [detPos[1] + np.sin(detPhi+tubeAng)*chordLen], s=200,
                   label='Collimator')
        
        ax.legend()
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Distance to detector (z={np.round(detPos[2], 2)}m)')
        
        fig.colorbar(pltObj2, ticks=np.arange(0, np.max(plottingDist), 0.1), 
                     label='[m]')
        
        plt.show()
    
        #### Plot the error evolution
        
        fig = plt.figure(figsize=(12, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        ax.plot(totParticles, errorArr, linewidth=3)

        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Error')

        ax.set_yscale('log')
        ax.set_xscale('log')

        plt.show()
        
    return threeDPoints, volumeWeights, volumeDistances, errorArr, totParticles

def on_segment(p,q,r):
    """
    This function checks if r lies on the line segment defined by (p,q).
    
    Code copied from- https://www.kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python
    which is why inline comments are sparse.

    Parameters
    ----------
    p : list
        Format- [xVal,yVal]
        One end of the line segment.
    q : list
        Format- [xVal,yVal]
        Other end of the line segment.
    r : list
        Format- [xVal,yVal]
        Point which is being checked.

    Returns
    -------
    boolean : True - r is on the line
              False- r is not on the line.    
    """

    if r[0]<=max(p[0],q[0]) and r[0]>=min(p[0],q[0]) and r[1]<=max(p[1],q[1]) and r[1]>=min(p[1],q[1]):
        return True
    
    return False

def orientation(p,q,r):
    """
    This function checks the orientation of the 3 points p,q,r. Are they 
    clockwise, counterclockwise or colinear with respect to each other.
    
    Code copied from- https://www.kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python
    which is why inline comments are sparse.

    Parameters
    ----------
    p : list
        Format- [xVal,yVal]
        Point 1.
    q : list
        Format- [xVal,yVal]
        Point 2.
    r : list
        Format- [xVal,yVal]
        Point 3.

    Returns
    -------
    int : 0 - colinear
          1 - clockwise
          2 - counterclockwise
    """

    val=((q[1]-p[1])*(r[0]-q[0]))-((q[0]-p[0])*(r[1]-q[1]))
    
    if val==0: 
        return 0
    
    return 1 if val>0 else -1

def line_intersection(seg1,seg2):
    """
    This function checks if 2 line segments intersect.
    
    Code copied from- https://www.kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python
    which is why inline comments are sparse.

    Parameters
    ----------
    seg1 : list
        Format- [[x1,y1],[x2,y2]]
        Points defining line segment 1.
    seg2 : list
        Format- [[x1,y1],[x2,y2]]
        Points defining line segment 2.

    Returns
    -------
    boolean : True - seg1,seg2 intersect
              False- seg2,seg2 do not intersect
    """

    #Get the 4 points that define the line segments
    p1,q1=seg1
    p2,q2=seg2

    #Find all possible oprientations
    o1=orientation(p1,q1,p2)
    o2=orientation(p1,q1,q2)
    o3=orientation(p2,q2,p1)
    o4=orientation(p2,q2,q1)

    #Check the general case
    if o1!=o2 and o3!=o4:
        return True

    #Check the special cases
    if o1==0 and on_segment(p1,q1,p2) : return True
    if o2==0 and on_segment(p1,q1,q2) : return True
    if o3==0 and on_segment(p2,q2,p1) : return True
    if o4==0 and on_segment(p2,q2,q1) : return True

    return False

def line_plane_intersection(p0,p1,p_co,p_no,epsilon=1e-6):
    """
    This function checks if a line intersects a plane.
    
    The line is defined by 2 points- p0,p1.
    The plane is defined by a vector normal to the plane p_n0 and a point on
    the plane p_co.
    
    Code is copied from- https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    which is why inline comments are sparse.

    Parameters
    ----------
    p0 : np.array
        Format- [xVal,yVal,zVal]
        One point that defines the line.
    p1 : np.array
        Format- [xVal,yVal,zVal]
        Other point that defines the line.
    p_co : np.array
        Format- [xVal,yVal,zVal]
        Point on the plane.
    p_no : np.array
        Format- [xVal,yVal,zVal]
        Vector normal to the plane.
    epsilon : np.array, optional
        Limit for how || the line can be to the plane. 
        The default is 1e-6.

    Returns
    -------
    np.array : [x,y,z] point of intersection if the line intersects the plane.
               0 if the line is || to the plane
    """

    u=p1-p0
    dot=np.dot(p_no,u)

    if abs(dot)>epsilon:
        
        w=p0-p_co
        fac=-np.dot(p_no,w)/dot
        u=u*fac
        
        return p0+u

    #Line segment is || to the plane.
    return 0

def close_to_line(point,lp1,lp2,epsilon=1e-4):
    """
    This function checks if a point is close to a line segment defined by lp1
    and lp2.
    
    Code is copied from- http://www.fundza.com/vectors/point2line/index.html
    which is why inline comments are sparse.

    Parameters
    ----------
    point : np.array
        Point to be checked.
    lp1 : np.array
        One end of the line segment.
    lp2 : np.array
        Other end of the line segment.
    epsilon : np.array
        Closeness parameter.
        The default value is 1e-4

    Returns
    -------
    bool
        Whether the point is close to the line segment.
    """
    
    line_vec=lp1-lp2
    pnt_vec=lp1-point
    line_len=np.sqrt(line_vec.dot(line_vec))
    line_unitvec=line_vec/line_len
    pnt_vec_scaled=pnt_vec*(1/line_len)
    t=np.dot(line_unitvec,pnt_vec_scaled)
    
    if t<0:
        t=0
    elif t>1:
        t=1
    
    nearest=line_vec*t
    dist=np.sqrt((nearest-pnt_vec).dot(nearest-pnt_vec))
    nearest=nearest+lp1
    
    if dist<=epsilon:
        return True
    else:
        return False

def close_to_detector(point,detSize,detPos):
    """
    This function checks if the point of intersection as calculated by
    line_plane_intersection is close to the detector. i.e. is a valid hit on
    the neutron detector.

    Parameters
    ----------
    point : np.array
        Point to be checked for closeness to the detector.
    detSize : float
        Size of the detector in m^3. Assume circular shape.
    detPos : np.array
        Position of the center of the detector.

    Returns
    -------
    bool
        Whether the point is close to the detector.
    """
    
    #Get the maximum allowable distance
    distLim=np.sqrt(detSize/np.pi)
    
    #Distance between the 2 points
    dist=np.sqrt((point-detPos).dot(point-detPos))
    
    if dist<=distLim:
        return True
    else:
        return False

def hit_detector(particleTrack, detPos, detNorm, detSize):
    """
    This function checks if the particle hits the detector.
    If the particle hits the detector, this function also returns the position
    and velocity at the time of the hit.

    Parameters
    ----------
    particleTrack : np.array
        Singple particle track data.
    detPos : np.array
        Position of the center of the detector.
    detNorm : np.array
        Normal vector of the detector.
    detSize : float
        Size of the detector

    Returns
    -------
    detectorHit : bool
        Whether the particle hit the detector.
        True- The particle hit the detector.
        False- The particle did not hit the detector.
    hitPos : np.array
        Position of the hit.
        Default is [0,0,0]
    hitVel : np.array
        Velocity of the hit.
        Default is [0,0,0]
    hitTrack : np.array
        Particle track upto the detector hit.
    """
    
    #Initialize the variables
    detectorHit=False
    hitPos=np.array([0,0,0])
    hitVel=np.array([0,0,0])
    hitTrack=[]
    
    #Go over the current track
    for j in range(len(particleTrack[0])-1):
        
        #Position of the particle
        point1=particleTrack[0:3,j]
        
        #Check if the track hits the wall
        if hit_vessel(point1)==True:
            #No need to track if the particle hits the vessel
            break
        
        #Next point to define a line segment to test intersection
        point2=particleTrack[0:3,j+1]
        
        #Check if the line intersects the plane
        intPoint=line_plane_intersection(point1,point2,detPos,detNorm)
        if type(intPoint)==np.ndarray:
            
            #Check if the point of intersection is close to the track and detector
            onLine=close_to_line(intPoint,point1,point2)
            onDetector=close_to_detector(intPoint,detSize,detPos)
            if onLine==True and onDetector==True:
                
                #Update the values
                detectorHit=True
                hitPos=intPoint
                hitVel=particleTrack[3:,j]
                
                #Only add the track up when it hits the detector
                hitTrack=particleTrack[:,j+1]
                
                #Stop tracking if the particle hits the detector
                break
    
    return detectorHit,hitPos,hitVel,hitTrack

def hit_vessel(point):
    """
    This function tests if the particle has hit the vessel wall.

    Parameters
    ----------
    point : np.array
        Current position of the particle.
        Format- [x,y,z] in meters

    Returns
    -------
    bool
        Whether the particle has hit the detector.
        True- Particle has hit the vessel wall.
        False- Particle has not hit the vessel wall.
    """
    
    #Radial position
    radPos=np.sqrt(point[0]**2+point[1]**2)
    
    #Radial boundary of the vessel
    if radPos>=0.45:
        return True
    #Axial boundary of the vessel
    if abs(point[2])>=1.44:
        return True
    
    #Base case
    return False

def through_the_core(particleTrack,coreRad):
    """
    This function checks if the particle went through the core of the plasma as
    defined by coreRad.
    If the particle went through the core, it also returns the particle track
    upto that point.

    Parameters
    ----------
    particleTrack : np.array
        Single particle track data.
    coreRad : float
        Radius of the core [m].

    Returns
    -------
    throughTheCore : bool
        Whether the particle went through the core.
        True- The particle went through the core.
        False- The particle did not go through the core.
    coreTrack : np.array
        Particle track data upto the point of leaving the core.
    """
    
    #Initialize the variables
    throughTheCore=False
    inCore=False
    coreTrack=[]
    
    #Go over the current track
    for j in range(len(particleTrack[0])-1):
        
        #Position of the particle
        point1=particleTrack[0:3,j]
        
        #Radial position of the particle
        radPos=np.sqrt(point1[0]**2+point1[1]**2)
        
        #Check if it enters the core
        if radPos<=coreRad:
            
            #Change flag state
            inCore=True
            
        #Check if it leaves the core
        if inCore==True:
            
            if radPos>=coreRad:
                
                #Only add the part until it leaves the core
                coreTrack=particleTrack[:,:j+1]
                
                #Update the state
                throughTheCore=True
                
                #No need to track further if particle leaves the core
                break
    
    return throughTheCore,coreTrack

def pad_array(arr):
    """
    This function pads all the sub arrays with nan values if arr is ragged.
    
    Code copied from-
    https://stackoverflow.com/questions/24494356/how-to-find-min-max-values-in-array-of-variable-length-arrays-with-numpy
    
    I added an if-else that checks if the subarrays are lists or np.arrays
    
    Parameters
    ----------
    arr : list
        Ragged list to be padded.

    Returns
    -------
    np.array
        Uniform array which has been padded with nan values.
    """
    
    #Get the maximum length of a subarray
    M = max(len(a) for a in arr)
    
    #Check if subarrays are lists or np.arrays
    if type(arr[0])==np.ndarray:
        return np.array([a.tolist() + [np.nan] * (M - len(a)) for a in arr])
    else:
        return np.array([a + [np.nan] * (M - len(a)) for a in arr])

def three_d_fusion_reactivity(filenameReactivity, threeDPoints, makeplot=False):
    """
    This function computes the fusion reacitivty at each voxel as specified by
    threeDPoints. Usually, threeDPoints is obtained from the output of the
    volume_weights function.
    
    fusionReactivity has units- # of particles / (m^3 * s)
    fusionReactivityVoxel has units- # of particles / (voxel * s)
    
    fusionReactivityVoxel should be used when trying to predict the absolute
    counts seen by the detectors.

    Parameters
    ----------
    filenameReactivity: string
        Name of the file that stores the 2D fusion reactivity profile.
        If it is a directory, it opens the 3 files associated with the .csv
        files. Else, it opens the .npz.
        
    threeDPoints : np.array
        A set of voxels that define the centers of the 3D grid which we are
        using to model the WHAM volume.
        Shape - 4D
        1st index- Voxel index number
        2nd index- x-position
        3rd index- y-position
        4th index- z-position
        
    makeplot : boolean
        Make a contour plot of the neutral reactivity profile as provided via
        CQL3D.
        Default - False

    Returns
    -------
    fusionReactivity : np.array
        Units = # of particles / (s m^3)
        Absolute fusion reactivity at each voxel.
        Shape - 1D where each index matches the 1st index of threeDPoints
        
    fusionReactivityVoxel : np.array
        Units = # of particles / (s voxel)
        Absolute fusion reactivity at each voxel.
        Shape - 1D where each index matches the 1st index of threeDPoints
    """
    
    #### Load the neutron rate from a simulation
    
    # Load data from the csv files
    if filenameReactivity[-1] == '/':
        
        # r values
        rArr = []
        with open(dataDest+'WHAM_r_2d_real.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                rArr.append(row)
        rArr = np.array(rArr)
        rArr = rArr.astype(np.float64)
    
        # z values
        zArr = []
        with open(dataDest+'WHAM_z_2d_real.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                zArr.append(row)
        zArr = np.array(zArr)
        zArr = zArr.astype(np.float64)
    
        # Fusion reactivity
        neutronRate = []
        with open(dataDest+'WHAM_neutron_rate_2d_real.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                neutronRate.append(row)
        neutronRate = np.array(neutronRate)
        neutronRate = neutronRate.astype(np.float64)
    
        # Interpolation function for the reactivity map
        interpolationFunction = sc.interpolate.RectBivariateSpline(rArr[:, 0], zArr[0], neutronRate)
    
    # Load from an .npz created from predicted_fusion_reactivity
    if filenameReactivity[-3:] == 'npz':
        
        dataObj = np.load(filenameReactivity)
        rArr2D = dataObj['rArr2D']
        zArr2D = dataObj['zArr2D']
        protonRate = dataObj['reactivity2D']
        
        rArrFlat = rArr2D.flatten()
        zArrFlat = zArr2D.flatten()
        protonRateFlat = protonRate.flatten()
        points = np.vstack((rArrFlat, zArrFlat)).T
        
        # Interpolation function for the reactivity map
        interpolationFunction = sc.interpolate.LinearNDInterpolator(points, protonRateFlat, fill_value=0)
    
    #### Get the fusion reactivity on each point as defined by threeDPoints
    
    fusionReactivity = np.zeros(shape=len(threeDPoints))

    rVals = np.sqrt(threeDPoints[:, 0]**2 + threeDPoints[:, 1]**2)
    zVals = np.abs(threeDPoints[:, 2])
    fusionReactivity = interpolationFunction(rVals, zVals)
    
    #### Fusion rate of each voxel
    uniqueX = np.sort(np.unique(threeDPoints[:, 0]))
    uniqueY = np.sort(np.unique(threeDPoints[:, 1]))
    uniqueZ = np.sort(np.unique(threeDPoints[:, 2]))
    voxelVol = (uniqueX[1] - uniqueX[0]) * (uniqueY[1] - uniqueY[0]) * (uniqueZ[1] - uniqueZ[0])
    fusionReactivityVoxel = fusionReactivity * voxelVol
    
    if makeplot == True:
        
        # x-z plane
        y0 = np.where(threeDPoints[:, 1] == -5.551115123125783e-17)
        y0 = np.where(threeDPoints[:, 1] == np.min(np.abs(threeDPoints[:, 1])))
        xzPoints = threeDPoints[y0]
        twoDReactivity = fusionReactivity[y0]

        fig = plt.figure(figsize=(17, 8), tight_layout='True')
        ax = fig.add_subplot(111)

        pltObj = ax.tricontourf(xzPoints[:, 2], xzPoints[:, 0], twoDReactivity, levels=100)
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('X [m]')
        ax.set_title('2D Fusion reactivity profile')
        cbar = fig.colorbar(pltObj, ticks = np.linspace(0, np.max(twoDReactivity), 5))

        plt.show()
    
    return fusionReactivity, fusionReactivityVoxel

def absolute_detector_response(filenameReactivity, detPos, detPhi, detSize, bendRad, tubeAng):
    """
    This function calculates the estimated counts/s for a given detector
    geometry. This is done by calculating the detector response function for a
    given magnetic equilibrium and detector geometry and then weighting it by
    the fusion reactivity profile.

    Parameters
    ----------
    filenameReactivity: string
        Name of the file that stores the 2D fusion reactivity profile.
        If it is a directory, it opens the 3 files associated with the .csv
        files. Else, it opens the .npz.
    detPos : np.array
        Position of the center of the detector.
        [x, y, z]
    detPhi : float
        Angle the detector makes with the x-axis in the XY plane. [radians]
    detSize : float
        Size of the detector (assume circular shape). [m^2]
    bendRad : float
        Bend radius of the collimating tube. [m]
    tubeAng : float
        Angle subtended by the collimating tube. [radians]

    Returns
    -------
    detectorRate : float
        Rate of hits on the detector in counts/s.
    """

    startTime = time.time()

    # Weights for each voxel
    threeDPoints, volumeWeights, volumeDistances, _, _ = volume_weights(detPos, detPhi, detSize, bendRad, tubeAng, 
                                                                        errorLim=1e-2,
                                                                        minParticles=200,
                                                                        maxParticles=2000,
                                                                        cellSize=2e-2,
                                                                        makeplot=False)
    
    weightsTime = time.time()
    print(f'It took {np.round(weightsTime-startTime, 2)}s to get the volume weights')
    print('Calculating the reactivity')

    # Fusion rate per voxel
    _, fusionReactivityVoxel = three_d_fusion_reactivity(filenameReactivity, threeDPoints)
    
    reactivityTime = time.time()
    print(f'It took {np.round(reactivityTime-weightsTime, 2)}s to calculate the fusion reactivity.')

    # Weighted fusion rate
    weightedRate = volumeWeights * fusionReactivityVoxel
    
    # Solid angle subtended by the detector to each voxel
    volumeDistances[volumeDistances == 0] = np.inf
    solidAngles = detSize / (4*np.pi*(volumeDistances**2))
    
    # Multiply the weighted rate by the solid angle
    measuredRate = solidAngles * weightedRate

    # Sum all the 'measured' rates to get the number of hits on the detector per second
    detectorRate = np.sum(measuredRate)
    
    return detectorRate

def b_field_interpolation(filenameEqdsk):
    """
    This function defines the magnetic field interpolation global variables
    used by a bunch of other functions in this script.

    Parameters
    ----------
    filenameEqdsk : str
        Location of the eqdsk file being used to generate the magnetic field
        interpolation variables.
    """
    
    print('Initializing magnetic field interpolation variables')

    global psilim, Rmesh, Zmesh, eqdsk_psi, BzInterpolator, BrInterpolator, BmagInterpolator
    
    # Start the script to create an .npz file with the quantities of interest
    # in the shared pleiades_env
    subprocess.run(['/share/envs/pleiades_env/bin/python',
                    'eqdsk_analysis_functions.py',
                    '-filename', filenameEqdsk], check=True)

    # Load the data from the .npz file made by this function
    npzFilename = np.load('filenameEqdsk'+'.npz', allow_pickle=True)

    # These are all 2D arrays with regular spacing for r and z.
    Rmesh = npzFilename['Rmesh']
    Zmesh = npzFilename['Zmesh']
    Br = npzFilename['Br']
    Bz = npzFilename['Bz']
    Bmag = npzFilename['Bmag']
    eqdsk_psi = npzFilename['magneticFlux']
    psilim = npzFilename['psilim']

    r1D = Rmesh[0]
    z1D = Zmesh[:, 0]

    # Interpolation function for magnetic field data
    # Converts the grid data into a function which can return Br and Bz at any
    # given (r,z)
    BzInterpolator = sc.interpolate.RectBivariateSpline(z1D, r1D, Bz)
    BrInterpolator = sc.interpolate.RectBivariateSpline(z1D, r1D, Br)
    BmagInterpolator = sc.interpolate.RectBivariateSpline(z1D, r1D, Bmag)

    return

def compute_row_detector(i, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr):
    """
    Used to calculate the detector response in parallel. This is purely a
    helper function used by generate_detector_response to run things in
    parallel.
    """

    detResponse = absolute_detector_response(filenameReactivity,
                                             detPos=detPosArr[i],
                                             detPhi=detPhiArr[i],
                                             detSize=detSizeArr[i],
                                             bendRad=bendRadArr[i],
                                             tubeAng=tubeAngArr[i])
    
    return i, detResponse

def generate_detector_response(filenameEqdsk, filenameReactivity,
                               detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr, makeplot=False, savename=None):
    """
    This function generates the detector response for a given set of-
    1. Magnetic equilibirum (as given in filenameEqdsk)
    2. Fusion reactivity profile (as given in filenameReactivity)
    3. Detector geoemtry (as specified by all the other parameters)
    
    detResponse is the predicted rate of each detector in counts/s

    Parameters
    ----------
    filenameEqdsk : str
        Location of the eqdsk file being used to generate the magnetic field
        interpolation variables.
    filenameReactivity: string
        Name of the file that stores the 2D fusion reactivity profile.
        If it is a directory, it opens the 3 files associated with the .csv
        files. Else, it opens the .npz.
    detPos : np.array
        2D array with the positions of the center of the detectors.
        1st index - Detector number
        2nd index - [x, y, z]
    detPhiArr : np.array
        Angle the detector makes with the x-axis in the XY plane. [radians]
    detSizeArr : np.array
        Size of the detector (assume circular shape). [m^2]
    bendRadArr : np.array
        Bend radius of the collimating tube. [m]
    tubeAngArr : np.array
        Angle subtended by the collimating tube. [radians]
    makeplot : boolean, optional
        Make a plot of the detector response function. The default is False.
    savename : str, optional
        Name of the file to save the detector response as a .npz file. The default is None.

    Returns
    -------
    detResponseArr : np.array
        Predicted measurement rate for each detector in counts/s.
    """
    
    # Create the magnetic field global variables
    b_field_interpolation(filenameEqdsk)

    # Initialize
    detResponseArr = np.zeros_like(detSizeArr)
    
    # Setup so that compute_row_detector can only be called with i
    worker = partial(compute_row_detector,
                     filenameReactivity=filenameReactivity,
                     detPosArr=detPosArr,
                     detPhiArr=detPhiArr,
                     detSizeArr=detSizeArr,
                     bendRadArr=bendRadArr,
                     tubeAngArr=tubeAngArr)
    
    # Run each detector in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(worker, range(len(detResponseArr)))

        for i, detResponse in results:
            detResponseArr[i] = detResponse
    
    # # Go over each detector and calculate its detector response
    # for i in range(len(detResponseArr)):
        
    #     print(f'Detector {i+1} of {len(detResponseArr)}')
    #     startT = time.time()
        
    #     detResponseArr[i] = absolute_detector_response(filenameReactivity,
    #                                                    detPosArr[i], 
    #                                                    detPhiArr[i], 
    #                                                    detSizeArr[i], 
    #                                                    bendRadArr[i], 
    #                                                    tubeAngArr[i])
        
    #     stopT = time.time()
    #     print(f'Generation time = {np.round(stopT-startT, 2)}s')
        
    if makeplot == True:
        
        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)
        
        ax.plot(detPosArr[:, 2], detResponseArr/1e3, linewidth=3)
        ax.scatter(detPosArr[:, 2], detResponseArr/1e3, s=200)
        
        ax.set_ylim(0, None)
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('Response [counts/ms]')
        
        plt.show()
    
    if savename is not None:
        np.savez('/home/sanwalka/synthetic_proton_detector/reactivity/'+savename, 
                 detResponseArr=detResponseArr/1e3,
                 detPosArr=detPosArr)
    
    return detResponseArr

def detector_angle_optimization(detPos, detSize, bendRad, tubeAng, filenameEqdsk, makeplot=False):
    """
    This function generates particle tracks for a set of detector angles and finds the best one.

    Here, best is defined as the angle which minimizes the impact parameter of each particle track.
    i.e. The angle that gives the best view into the core of the plasma

    Parameters
    ----------
    detPos : np.array
        Position of the center of the detector. [m]
        [x, y, z]
    detSize : float
        Size of the detector (assume circular shape). [m^2]
    bendRad : float
        Bend radius of the collimating tube. [m]
    tubeAng : float
        Angle subtended by the collimating tube. [radians]
    filenameEqdsk : string
        Name/location of the eqdsk file.

    Returns
    -------
    optimalAngle : float
        Optimal angle for the detector to view the core plasma.
        Units - deg
    """

    # Angle array
    angleArr = np.arange(250, 310, 2)
    angleArrRad = (np.pi/180) * angleArr
    # Impact parameter array
    minImpactParams = np.zeros_like(angleArr, dtype=float)

    # Generate the magnetic field variables
    b_field_interpolation(filenameEqdsk)

    # Go over each angle and generate the particle tracks
    for i in range(len(angleArr)):

        currTracks = generate_tracks_aperture(detPos = detPos, 
                                              detPhi = angleArrRad[i], 
                                              detSize = detSize, 
                                              bendRad = bendRad, 
                                              tubeAng = tubeAng)

        # Go over each track and get the impact parameter
        impactParams = np.zeros(shape=len(currTracks))

        for j in range(len(currTracks)):

            # xy-positions
            xPosArr = currTracks[j][0]
            yPosArr = currTracks[j][1]

            rPosArr = np.sqrt(xPosArr**2 + yPosArr**2)

            impactParams[j] = np.min(rPosArr)

        minImpactParams[i] = np.average(impactParams)

    # Set all nan values to a large number
    minImpactParams = np.nan_to_num(minImpactParams, nan=0.1)

    # Calculate the optimal angle
    minIdx = np.argmin(minImpactParams)
    optimalAngle = int(angleArr[minIdx])

    if makeplot:

        fig = plt.figure(figsize=(12, 8), tight_layout=True)
        ax = fig.add_subplot(111)

        ax.plot(angleArr, minImpactParams)

        ax.set_xlabel('Detector angle [deg]')
        ax.set_ylabel('Impact Parameter [m]')

        plt.show()

    return optimalAngle

# Plot normalized detector response function
if __name__ == '__tempmain__':
    """
    Used to generate and plot the normalized detector response function for a given magnetic equilibrium and detector geometry. 
    
    This is useful for visualizing the response function and making sure it looks reasonable.
    """

    detPos = np.array([-0.257,  0.307,  0.5])
    detPhi = (np.pi/180) * (280)
    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550 # m^2
    bendRad = 0.7 # meters
    tubeAng = 10 * np.pi/180 # radians

    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    b_field_interpolation(filenameEqdsk)

    _, _, _, _, _ = volume_weights(detPos, detPhi, detSize, bendRad, tubeAng, 
                                   cellSize=1e-2, errorLim=None, maxParticles=500,
                                   makeplot=True, savename='volume_weights_0.7m_10deg')

# Save normalized detector response function
if __name__ == '__tempmain__':
    """
    Used to generate the detector response for a given magnetic equilibrium, 
    fusion reactivity profile and detector geometry. 
    
    The output is saved as a .npz file in the reactivity folder.
    """

    ##############################################################################
    #### Detector positions (boxport)

    # The detectors are evenly spaced on the boxport
    zPosArr = np.arange(0.362, 0.665, 2*2.54/1e2) # 2 in apart on the boxport
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters
    
    # Detector angles
    detPhiArr = np.array([282, 278, 275, 271, 264, 258]) * (np.pi/180)
    ##############################################################################

    ##############################################################################
    #### Detector positions (HTPD 2026)
    zPosArr = np.arange(0.2, 0.61, 0.05)
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

    # Detector angles
    detPhiArr = np.array([296, 294, 292, 290, 288, 286, 284, 278, 273]) * (np.pi/180)
    ##############################################################################

    # Detector sizes
    detRad = 0.5 # inches
    detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

    # Tube bend radii
    bendRadArr = np.full(len(zPosArr), 1.2) # meters
    
    # Tube sector angle
    tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians
    
    #### eqdsk file
    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    
    #### Fusion reactivity profile
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/Te_2keV_NBI_2d_reactivity.npz'
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d.npz'
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_with_maxwellian.npz'
    
    """
    Generate the detector response for the given-
    1. Magnetic equilibrium
    2. Reactivity profile
    3. Detector geometry
    """
    
    detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                             makeplot=True, savename='detector_response_with_maxwellian.npz')
    
# Plot the detector array response for a given reactivity profile
if __name__ == '__main__':
    """
    Calculate and plot the detector response for a given reactivity profile and detector geometry.
    """

    ##############################################################################
    # Detector geometry

    # (x,y,z) positions. These are mounted on the CC inner wall.
    zPosArr = np.arange(0.2, 0.61, 0.05)
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

    # Detector angles (after optimization)
    detPhiArr = np.array([292, 292, 288, 288, 286, 284, 280, 276, 268]) * (np.pi/180) # radians

    # Detector sizes
    detRad = 0.5 # inches
    detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

    # Tube bend radii
    bendRadArr = np.full(len(zPosArr), 1.2) # meters
    
    # Tube sector angles
    tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians
    ##############################################################################

    # 2D reactivity profile
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_10percent_fast_ions.npz'

    # eqdsk file
    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    # Generate the detector response
    detResponse = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                             makeplot=True, savename='detector_response_with_10percent_fast_ions.npz')

# Check effect of different distribution functions
if __name__ == '__tempmain__':
    """
    Check effect of maxwellian and fast ion components on the detector response.
    """

    ##############################################################################
    #### Detector positions (HTPD 2026)
    zPosArr = np.arange(0.2, 0.61, 0.05)
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters

    # Detector angles
    detPhiArr = np.array([292, 292, 288, 288, 286, 284, 280, 276, 268]) * (np.pi/180)
    ##############################################################################

    # Detector sizes
    detRad = 0.5 # inches
    detSizeArr = np.full(len(zPosArr), (np.pi*detRad*detRad) / 1550) # m^2

    # Tube bend radii
    bendRadArr = np.full(len(zPosArr), 1.2) # meters
    
    # Tube sector angle
    tubeAngArr = np.full(len(zPosArr), 10 * np.pi/180) # radians
    
    #### eqdsk file
    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    # Fast and maxwellian
    print('Calculating response with both fast and maxwellian components')
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_fast_and_maxwellian.npz'
    detResponseFastAndMaxwellian = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                              makeplot=False)

    # Only maxwellian
    print('Calculating response with only maxwellian component')
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_maxwellian_only.npz'
    detResponseMaxwellianOnly = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                           makeplot=False)

    # Only fast
    print('Calculating response with only fast ion component')
    filenameReactivity = '/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_fast_only.npz'
    detResponseFastOnly = generate_detector_response(filenameEqdsk, filenameReactivity, detPosArr, detPhiArr, detSizeArr, bendRadArr, tubeAngArr,
                                                     makeplot=False)
    
    # Plot the 3 responses together
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # Fast and Maxwellian
    ax.plot(detPosArr[:, 2], detResponseFastAndMaxwellian/1e3, linewidth=3, label='Fast + Maxwellian')
    ax.scatter(detPosArr[:, 2], detResponseFastAndMaxwellian/1e3, s=200)
    
    # Maxwellian only
    ax.plot(detPosArr[:, 2], detResponseMaxwellianOnly/1e3, linewidth=3, label='Maxwellian only')
    ax.scatter(detPosArr[:, 2], detResponseMaxwellianOnly/1e3, s=200)
    
    # Fast only
    ax.plot(detPosArr[:, 2], detResponseFastOnly/1e3, linewidth=3, label='Fast only')
    ax.scatter(detPosArr[:, 2], detResponseFastOnly/1e3, s=200)
    
    ax.legend()
    ax.set_ylim(0, None)
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('Response [counts/ms]')
    
    plt.show()

# Check particle tracks
if __name__ == '__tempmain__':
    """
    Used to check the angle of a specific detector to make sure it sees the core of the plasma.
    """

    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'
    
    b_field_interpolation(filenameEqdsk)
    
    detPos = np.array([-0.257,  0.307,  0.5])
    
    detPhi = (np.pi/180) * (250)
    
    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550
    
    bendRad = 0.7
    
    tubeAng = 10 * np.pi/180

    # Save the particle tracks
    savename = '/home/sanwalka/synthetic_proton_detector/particle_tracks/track_data_0.7m_10deg.pkl'
    
    openingTracks = generate_tracks_aperture(detPos, detPhi, detSize, bendRad, tubeAng, makeplot=True, saveplot=False, savename=savename)

# Find the optimal angle for a given detector position
if __name__ == '__tempmain__':

    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    detPos = np.array([-0.257,  0.307,  0.5])
    
    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550
    
    bendRad = 0.7
    
    tubeAng = 10 * np.pi/180

    optimalAngle = detector_angle_optimization(detPos, detSize, bendRad, tubeAng, filenameEqdsk, makeplot=True)

# Find the optimal angle for all of the detectors
if __name__ == '__tempmain__':

    filenameEqdsk='/home/sanwalka/synthetic_proton_detector/eqdsk/wham_hts_eqdsk_for_kunal'

    zPosArr = np.arange(0.2, 0.61, 0.05)
    xPos = -0.257
    yPos = 0.307
    detPosArr = np.array([[xPos, yPos, zPos] for zPos in zPosArr]) # meters
    
    detRad = 0.5 # inches
    detSize = (np.pi*detRad*detRad) / 1550
    
    bendRad = 0.7
    
    tubeAng = 10 * np.pi/180

    optimalAngleArr = np.zeros_like(zPosArr)

    for i in range(len(detPosArr)):

        print(f'Optimizing detector {i+1} of {len(detPosArr)}')

        optimalAngleArr[i] = detector_angle_optimization(detPosArr[i], detSize, bendRad, tubeAng, filenameEqdsk)

    np.savez(file = '/home/sanwalka/synthetic_proton_detector/data/optimal_angles_5cm_spacing.npz',
             zPosArr = zPosArr,
             optimalAngleArr = optimalAngleArr)

    # Plot the optimal angles
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.scatter(zPosArr, optimalAngleArr, s=200)
    ax.plot(zPosArr, optimalAngleArr, linewidth=3)

    ax.set_title('Optimal Angles for detectors')
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('Angle [deg]')

    plt.show()