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
import pickle
import numpy as np
import scipy as sc
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from random_geometry_points.plane import Plane
from multiprocessing import Pool, cpu_count
from functools import partial

from geometry_utils import hit_detector, through_the_core

plt.rcParams.update({'font.size': 22})
plt.switch_backend('TkAgg')

global plotDir, reactivityDir
plotDir = '/home/sanwalka/synthetic_proton_detector/plots/'
reactivityDir = '/home/sanwalka/synthetic_proton_detector/reactivity/'

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

def batch_particle_tracks(xIniArr, energy, thetaArr, phiArr, species='H',
                          timesteps=50, steplength=1e-9):
    """
    This function tracks the motion of a whole batch of particles through the
    magnetic field at once, given their initial positions and velocities.

    This is the vectorized form of single_particle_track: instead of a
    Python loop calling the B-field interpolators once per particle per
    timestep, it steps every particle together and calls the interpolators
    once per timestep for the whole batch (via grid=False evaluation). The
    physics (relativistic Boris pusher, constant gamma) is identical to
    single_particle_track, just batched — trajectories match it bit-for-bit.

    Parameters
    ----------
    xIniArr : np.array
        Initial positions of the particles.
        Shape- (N, 3)
    energy : float
        Energy of the particles (eV). Shared across the batch.
    thetaArr : np.array
        Angle with the z-axis for each particle (radians). Shape- (N,)
    phiArr : np.array
        Angle with the x-axis for each particle (radians). Shape- (N,)
    species : string, optional
        Species code.
        The default is 'H'.
    timesteps : int
        Number of timesteps.
    steplength : float
        Size of each timestep (seconds)

    Returns
    -------
    stateVecs : np.array
        State vector as a function of time for every particle.
        Shape- (N, 6, timesteps), stacking [x,y,z,vx,vy,vz] per particle -
        the same layout as np.array([single_particle_track(...), ...]).
    """

    xIniArr = np.asarray(xIniArr, dtype=float)
    thetaArr = np.asarray(thetaArr, dtype=float)
    phiArr = np.asarray(phiArr, dtype=float)
    numParticles = xIniArr.shape[0]

    # Initial velocity of every particle in cartesian coordinates
    vIniArr = initialize_particle(energy, thetaArr, phi=phiArr, species=species).T

    # Gamma is constant per particle: Lorentz force does no work in a pure magnetic field
    speed = np.linalg.norm(vIniArr, axis=1)
    gamma = 1.0 / np.sqrt(1.0 - (speed**2 / const.c**2))

    # Effective charge-to-mass ratio (relativistic), per particle
    qm_gamma = charge_to_mass_ratio(species) / gamma

    # Pre-allocate output arrays (time-major for cheap per-step writes)
    positions  = np.empty((timesteps, numParticles, 3))
    velocities = np.empty((timesteps, numParticles, 3))

    x = xIniArr.copy()
    v = vIniArr.copy()

    positions[0]  = x
    velocities[0] = v

    dt = steplength
    half_qm_gamma_dt = 0.5 * qm_gamma * dt  # (N,)

    # Particles still inside the interpolation domain
    active = np.ones(numParticles, dtype=bool)

    for i in range(1, timesteps):

        r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
        z = x[:, 2]

        # Outside interpolation domain — particle has left the machine.
        # Freeze its position and zero its velocity from here on, exactly
        # like the per-particle break in single_particle_track.
        newlyOut = active & ((r > 0.45) | (np.abs(z) > 1.0))
        if np.any(newlyOut):
            v[newlyOut] = 0.0
            active = active & ~newlyOut

        if np.any(active):

            Br = BrInterpolator(z, r, grid=False)
            Bz = BzInterpolator(z, r, grid=False)

            with np.errstate(divide='ignore', invalid='ignore'):
                inv_r = np.where(r > 0.0, 1.0 / r, 0.0)
            Bx = Br * x[:, 0] * inv_r
            By = Br * x[:, 1] * inv_r

            # Boris rotation step (no electric field)
            tx = half_qm_gamma_dt * Bx
            ty = half_qm_gamma_dt * By
            tz = half_qm_gamma_dt * Bz
            t_dot = tx*tx + ty*ty + tz*tz
            sx = 2.0 * tx / (1.0 + t_dot)
            sy = 2.0 * ty / (1.0 + t_dot)
            sz = 2.0 * tz / (1.0 + t_dot)

            # v' = v + v × t
            vpx = v[:, 0] + v[:, 1]*tz - v[:, 2]*ty
            vpy = v[:, 1] + v[:, 2]*tx - v[:, 0]*tz
            vpz = v[:, 2] + v[:, 0]*ty - v[:, 1]*tx

            # v_new = v + v' × s
            vNew = np.empty_like(v)
            vNew[:, 0] = v[:, 0] + vpy*sz - vpz*sy
            vNew[:, 1] = v[:, 1] + vpz*sx - vpx*sz
            vNew[:, 2] = v[:, 2] + vpx*sy - vpy*sx

            xNew = x + vNew * dt

            v[active] = vNew[active]
            x[active] = xNew[active]

        positions[i]  = x
        velocities[i] = v

        if not np.any(active):
            positions[i+1:]  = x[np.newaxis, :, :]
            velocities[i+1:] = v[np.newaxis, :, :]
            break

    # (timesteps, N, 3) -> (N, 3, timesteps), then stack pos/vel -> (N, 6, timesteps)
    return np.concatenate([positions.transpose(1, 2, 0), velocities.transpose(1, 2, 0)], axis=1)

def single_particle_track(xIni, energy, theta, phi=0, species='H',
                          timesteps=50, steplength=1e-9):
    """
    This function tracks the motion of the particle through the magnetic field
    as a function of the inital position and velocity.

    Convenience wrapper around batch_particle_tracks for a single particle -
    see that function for the actual (relativistic Boris pusher) integration.

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

    stateVecs = batch_particle_tracks(np.asarray([xIni]), energy,
                                      np.asarray([theta]), np.asarray([phi]),
                                      species=species, timesteps=timesteps,
                                      steplength=steplength)

    return stateVecs[0]

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

    # Launch position for every particle: numLaunchesPerPos particles per detector point
    xIniArr = np.repeat(planePoints, numLaunchesPerPos, axis=0)

    setupTime = time.time()
    # print('Time taken to set up track generation- '+str(setupTime-startTime)+' seconds')

    # Track every particle at once (vectorized Boris pusher)
    particleTracks = batch_particle_tracks(xIniArr, energy, thetaLaunchArr, phiLaunchArr, species=species)
    
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

def _track_grid_hits(track, x0, y0, z0, cellSize, zSpacing, Nx, Ny, Nz):
    """
    Finds which voxels of the regular (x0,y0,z0)-anchored grid used by
    volume_weights a single particle track passes through, and the
    cumulative distance travelled along the track to the first arrival at
    each voxel.

    Since the grid is regular, the voxel a given track position falls into
    can be computed directly by index arithmetic instead of comparing that
    position against every grid point -- this replaces what used to be an
    O(num_grid_points) brute-force distance check per track (the main cost
    of volume_weights) with an O(1)-per-position lookup.

    The "hit" tolerance replicates the original brute-force check exactly:
    a track position counts as hitting a voxel if it's within half a cell
    in x, y AND z. Since the z-grid spacing (zSpacing) is fixed at 2cm
    regardless of cellSize, that tolerance is 0.5*cellSize in z too -- not
    0.5*zSpacing -- so z-matching is stricter (or looser) than x/y whenever
    cellSize != zSpacing, exactly as before.

    Parameters
    ----------
    track : np.array
        Single particle track (shape (6, num_positions), as returned by
        generate_tracks_aperture).
    x0, y0, z0 : float
        Coordinates of the grid's first point (xArr[0], yArr[0], zArr[0]).
    cellSize : float
        Grid spacing in x and y, and the hit tolerance in all 3 dimensions.
    zSpacing : float
        Grid spacing in z.
    Nx, Ny, Nz : int
        Number of grid points in x, y, z.

    Returns
    -------
    hitIdx : np.array of int
        Linear indices (into volumeWeights/volumeDistances/threeDPoints) of
        the voxels this track hit, each appearing once.
    hitDist : np.array of float
        Cumulative track distance at the first arrival at each of those
        voxels.
    """

    positions = track[:3].T  # (numPositions, 3)
    numPositions = positions.shape[0]

    if numPositions > 1:
        deltas = np.diff(positions, axis=0)
        segmentLengths = np.linalg.norm(deltas, axis=1)
        cumulativeDist = np.concatenate([[0.0], np.cumsum(segmentLengths)])
    else:
        cumulativeDist = np.array([0.0])

    ixf = (positions[:, 0] - x0) / cellSize
    iyf = (positions[:, 1] - y0) / cellSize
    izf = (positions[:, 2] - z0) / zSpacing

    ix = np.round(ixf).astype(int)
    iy = np.round(iyf).astype(int)
    iz = np.round(izf).astype(int)

    # x/y always satisfy the tolerance once rounded, since grid spacing
    # there equals cellSize; z needs an explicit check (see docstring).
    zTolInCells = 0.5 * cellSize / zSpacing

    valid = ((ix >= 0) & (ix < Nx) &
            (iy >= 0) & (iy < Ny) &
            (iz >= 0) & (iz < Nz) &
            (np.abs(izf - iz) <= zTolInCells))

    if not np.any(valid):
        return np.array([], dtype=int), np.array([])

    # Linear index matching the (Ny,Nx,Nz) C-order flattening produced by
    # np.vstack(np.meshgrid(xArr, yArr, zArr)).reshape(3,-1).T in volume_weights
    linIdx = iy[valid] * (Nx * Nz) + ix[valid] * Nz + iz[valid]
    timeIdx = np.nonzero(valid)[0]

    # Keep only the first (earliest-time) arrival at each voxel
    uniqIdx, firstPos = np.unique(linIdx, return_index=True)
    firstTimeIdx = timeIdx[firstPos]

    return uniqIdx, cumulativeDist[firstTimeIdx]

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
    zSpacing = 2e-2
    zArr = np.arange(-1, 1+zSpacing, zSpacing)

    # Create an array of the 3D points
    # 1st index- point number
    # 2nd index- [x,y,z]
    threeDPoints = np.vstack(np.meshgrid(xArr, yArr ,zArr)).reshape(3,-1).T

    # Grid shape/origin, used by _track_grid_hits to index straight into
    # threeDPoints instead of comparing every track position against every
    # grid point.
    Nx, Ny, Nz = len(xArr), len(yArr), len(zArr)
    x0, y0, z0 = xArr[0], yArr[0], zArr[0]

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

    while True:

        # If errorLim is None, then we only stop when maxParticles is reached
        if errorLim != None:
            keepGoing = (currError >= errorLim and totParticles[-1] <= maxParticles) or totParticles[-1] <= minParticles
        else:
            keepGoing = totParticles[-1] <= maxParticles

        if not keepGoing:
            break

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
        for track in openingTracks:

            hitIdx, hitDist = _track_grid_hits(track, x0, y0, z0, cellSize, zSpacing, Nx, Ny, Nz)

            volumeWeights[hitIdx] += 1
            volumeDistances[hitIdx] = hitDist

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
        
        dataObj = np.load(reactivityDir+filenameReactivity)
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
                                                                        errorLim=None,
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
    # Resolved relative to this file (not the caller's cwd) so it still works
    # when called from driver scripts living outside the root directory.
    eqdskScriptPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eqdsk_analysis_functions.py')
    subprocess.run(['/share/envs/pleiades_env/bin/python',
                    eqdskScriptPath,
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