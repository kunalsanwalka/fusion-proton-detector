# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 12:29:11 2021

@author: kunal

This program contains a list of functions used to analyze eqdsk files which
store data about the magnetic fields.
"""

import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.interpolate import RegularGridInterpolator
from jesse_eqdsk_io import *
plt.rcParams.update({'font.size': 32})
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# Destination of the plots
# =============================================================================
plotDest='C:/Users/kunal/OneDrive - UW-Madison/WHAM/Plots/'

def parseArgs():
    """
    Allows for the eqdsk file being processed to be directly fed into this code via the terminal.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('-filename', type=str)

    args = parser.parse_args()

    return args

def read_eqdsk(filename):
    """
    This function was written by Ethan Peterson and it has been copied here 
    line for line.
    
    line 1 -- read grid information: title cursign, nnr, nnz, nnv
    line 2 -- read rbox,zbox,0,0,0
    line 3 -- read 0,0,0,psi_lim,0
    line 4 -- read total toroidal current,0,0,0,0
    line 5 -- read 0,0,0,0,0
    line 6 -- read list of toroidal flux for each flux surface (zeros)
    line 7 -- read list of pressure for each flux surface (zeros)
    line 8 -- read list of (RBphi)' for each flux surface (zeros)
    line 9 -- read list of P' for each flux surface (zeros)
    line 10 -- read flattened list of psi values on whole grid (NOT ZERO)
    line 11 -- read list of q for each flux surface (zeros)
    line 12 -- read number of coordinate pairs for limit surface and vessel surface
    line 13 -- read list of R,Z pairs for limiter surface, then vessel surface
    
    Parameters
    ----------
    filename : string
        Location of the eqdsk output file.
        
    Returns
    -------
    eq_dict : dictionary
        Object with all the relevant data from the eqdsk file.
    """
    eq_dict = {}
    
    try:
        with open(filename,"r") as f:
            lines = f.readlines()
            line1 = lines[0]
            # print(line1)
            eq_dict["title"] = line1[0:26]
            line1rem = line1[48:]
            # print(line1rem)
            eq_dict["cursign"] = int(line1rem.split()[-4])
            eq_dict["nnr"] = int(line1rem.split()[-3])
            eq_dict["nnz"] = int(line1rem.split()[-2])
            eq_dict["nnv"] = int(line1rem.split()[-1])
            line2 = lines[1].split()
            eq_dict["rbox"] = float(line2[-5])
            eq_dict["zbox"] = float(line2[-4])
            line3 = lines[2].split()
            eq_dict["psi_lim"] = float(line3[-2])
            line4 = lines[3].split()
            eq_dict["Ip"] = float(line4[-5])
            line5 = lines[4].split()
            fs_lines = int(np.ceil(eq_dict["nnv"]/5.0))
            head = [line.strip().split() for line in lines[5:5+fs_lines]]
            tor_flux = np.array([float(num) for line in head for num in line])
            eq_dict["tor_flux"] = tor_flux
            head = [line.strip().split() for line in lines[5+fs_lines:5+2*fs_lines]]
            p_flux = np.array([float(num) for line in head for num in line])
            eq_dict["p_flux"] = p_flux
            head = [line.strip().split() for line in lines[5+2*fs_lines:5+3*fs_lines]]
            rbphi_flux = np.array([float(num) for line in head for num in line])
            eq_dict["rbphi_flux"] = rbphi_flux
            head = [line.strip().split() for line in lines[5+3*fs_lines:5+4*fs_lines]]
            pprime_flux = np.array([float(num) for line in head for num in line])
            eq_dict["pprime_flux"] = pprime_flux
            # Read psi on whole grid, nnr x nnz
            nnr,nnz = eq_dict["nnr"],eq_dict["nnz"]
            rz_pts = nnr*nnz
            l0 = 5+4*fs_lines
            psi_lines = int(np.ceil(rz_pts/5.0))
            head = [line.strip().split() for line in lines[l0:l0+psi_lines]]
            psi = np.array([float(num) for line in head for num in line])
            eq_dict["psi"] = psi.reshape((nnz,nnr))
            rbox,zbox = eq_dict["rbox"],eq_dict["zbox"]
            # Kunal S. - Changed -zbox/2 to 0 and -zbox/2 to zbox
            # R,Z = np.meshgrid(np.linspace(0,rbox,nnr),np.linspace(-zbox/2,zbox/2,nnz))
            R,Z = np.meshgrid(np.linspace(0,rbox,nnr),np.linspace(0,zbox,nnz))
            eq_dict["R"] = R
            eq_dict["Z"] = Z
            head = [line.strip().split() for line in lines[l0+psi_lines:l0+psi_lines+fs_lines]]
            q_flux = np.array([float(num) for line in head for num in line])
            eq_dict["q_flux"] = q_flux
            nlim_pairs, nves_pairs = [int(x) for x in lines[l0+psi_lines+fs_lines].strip().split()]
            eq_dict["nlim_pairs"] = nlim_pairs
            eq_dict["nves_pairs"] = nves_pairs
            pair_lines = int(np.ceil((nlim_pairs + nves_pairs)*2.0/5.0))
            rest = [line.rstrip() for line in lines[l0+psi_lines+fs_lines+1:]]
            rest = [line[i:i+16] for line in rest for i in range(0,len(line),16)]
            pairs = np.array([float(num.strip()) for num in rest])
            lim_pairs = np.array(list(zip(pairs[0:2*nlim_pairs:2],pairs[1:2*nlim_pairs:2])))
            ves_pairs = np.array(list(zip(pairs[2*nlim_pairs::2],pairs[2*nlim_pairs+1::2])))
            eq_dict["lim_pairs"] = lim_pairs
            eq_dict["ves_pairs"] = ves_pairs
        
    except:
            
#        print('Using new reader')
        
        eqObj = EQDSK(filename)
        
        # 2d magnetic flux data
        eq_dict['psi'] = eqObj.psirz
        
        # r and z arrays
        nr = eqObj.nr
        nz = eqObj.nz
        rdim = eqObj.rdim
        zdim = eqObj.zdim
        
        r = np.linspace(0, rdim, nr)
        z = np.linspace(-zdim, zdim, nz)
        
        R, Z = np.meshgrid(r, z)
        
        eq_dict['R'] = R
        eq_dict['Z'] = Z / 2
        
        # limiter flux
        rlim = eqObj.rlim
        zlim = eqObj.zlim
        
        # Create an interpolation function for the flux        
        from scipy.interpolate import RectBivariateSpline
        interpolationFunction = RectBivariateSpline(Z[:, 0], R[0], eqObj.psirz)
        rlim = eqObj.rlim
        zlim = eqObj.zlim
        eq_dict['psi_lim'] = interpolationFunction(zlim[100], rlim[100])[0][0]

    return eq_dict
    
def magnetic_field_RZ(filename,makeplot=False,saveplot=False):
    """
    This function returns the magnetic field profile along with the associated
    coordinate arrays.
    
    NOTE- The field values are only for r>0. If you are making a XZ-plot. Refer
    to the plots in the function to see how to handle that case.
    
    To get the 1D from Rmesh and and Zmesh, use the following code-
    r1d=Rmesh[0]
    z1d=Zmesh[:,0]

    Parameters
    ----------
    filename : string
        Location of the eqdsk output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Rmesh : np.array
        R position.
    Zmesh : np.array
        Z position.
    Br : np.array
        Radial component of B.
    Bz : np.array
        Axial component of B.
    Bmag : np.array
        Magnetic field strength.
    """
    
    #Open the file
    eqDict=read_eqdsk(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=eqDict['R'][0]
    
    #z-array for equilibrium B
    eqdsk_z=eqDict['Z'][:,0]
    
    #Poloidal flux / 2*pi
    eqdsk_psi=eqDict['psi']
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    #dr and dz to take the gradient of the flux
    dr=eqdsk_r[1]-eqdsk_r[0]
    dz=eqdsk_z[1]-eqdsk_z[0]
    
    #Br and Bz
    Bz=np.gradient(eqdsk_psi,dr,axis=1)/(eqdsk_r)
    Br=-np.gradient(eqdsk_psi,dz,axis=0)/(eqdsk_r)
    
    #Correct for the / by 0 error when R=0
    Bz[:,0]=-Bz[:,2]+2*Bz[:,1]
    Br[:,0]=0
    
    #|B| (since we are axisymmetric, B_phi=0)
    Bmag=np.sqrt(Bz**2+Br**2)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savenameStrength=ncName+'_magnetic_field_strength.png'
        savenameBr=ncName+'_magnetic_field_r.png'
        savenameBz=ncName+'_magnetic_field_z.png'
        
        # =====================================================================
        # Magnetic Field Strength
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Bmag,levels=60)
        # ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Bmag,levels=60)
        # ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title('Magnetic Field Strength')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label('|B| [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameStrength,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # B_r
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Br,levels=60)
        # ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Br,levels=60)
        # ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title(r'Magnetic Field ($B_r$)')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$B_r$ [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameBr,bbox_inches='tight')
        plt.show()
        
        # =====================================================================
        # B_z
        # =====================================================================
        
        fig=plt.figure(figsize=(21,8))
        ax=fig.add_subplot(111)
        pltobj=ax.contourf(Zmesh,Rmesh,Bz,levels=60)
        # ax.contour(pltobj,colors='black')
        pltobj=ax.contourf(Zmesh,-Rmesh,Bz,levels=60)
        # ax.contour(pltobj,colors='black')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        ax.set_title(r'Magnetic Field ($B_z$)')
        ax.grid(True)
        cbar=fig.colorbar(pltobj)
        cbar.set_label(r'$B_z$ [T]')
        if saveplot==True:
            plt.savefig(plotDest+savenameBz,bbox_inches='tight')
        plt.show()
        
    return Rmesh,Zmesh,Br,Bz,Bmag

def Bmag_function(filename,r,z):
    """
    This function calculates the magnitude of the magnetic field at any given
    point

    Parameters
    ----------
    filename : string
        Location of the eqdsk output file.
    r, z : float
        Coordinates at which we want the magnetic field strength.

    Returns
    -------
    float
        Magnetic field strength [Tesla].
    """
    
    #Open the file
    eqDict=read_eqdsk(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=eqDict['R'][0]
    
    #z-array for equilibrium B
    eqdsk_z=eqDict['Z'][:,0]
    
    #Poloidal flux / 2*pi
    eqdsk_psi=eqDict['psi']
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    #dr and dz to take the gradient of the flux
    dr=eqdsk_r[1]-eqdsk_r[0]
    dz=eqdsk_z[1]-eqdsk_z[0]
    
    #Br and Bz
    Bz=np.gradient(eqdsk_psi,dr,axis=1)/(eqdsk_r)
    Br=-np.gradient(eqdsk_psi,dz,axis=0)/(eqdsk_r)
    
    #Correct for the / by 0 error when R=0
    Bz[:,0]=-Bz[:,2]+2*Bz[:,1]
    Br[:,0]=0
    
    #|B| (since we are axisymmetric, B_phi=0)
    Bmag=np.sqrt(Bz**2+Br**2)
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    return field_interpolator([z,r])[0]

def resonant_field_strengths(freq):
    """
    This function calculates the magentic field strength for the 1st 5
    harmonics of the D and T cyclotron frequency given the antenna frequency.

    Parameters
    ----------
    freq : Float
        Antenna frequency in Hz.

    Returns
    -------
    BDArr : np.array
        Deuterium |B| array.
    BTArr : np.array
        Tritium |B| array.
    """
    
    # =========================================================================
    # Magnetic fields for resonances
    # =========================================================================
    
    #Fundamental frequency field for D
    B0D=2*np.pi*2*constants.m_p*freq/constants.e
    
    #Fundamental frequency field for T
    B0T=2*np.pi*3*constants.m_p*freq/constants.e
    
    #Array with the 1st 9 harmonics
    BDArr=[]
    BTArr=[]
    for i in reversed(range(1,8)): #range() does not include the last number
        BDArr.append(B0D/i)
        BTArr.append(B0T/i)
    
    #Convert to numpy
    BDArr=np.array(BDArr)
    BTArr=np.array(BTArr)
    
    return BDArr,BTArr

def flux_surfaces(filename,makeplot=False,saveplot=False):
    """
    This function returns the flux surfaces from Pleiades as stored in the
    Genray .nc
    
    To get the 1D from Rmesh and and Zmesh, use the following code-
    r1d=Rmesh[0]
    z1d=Zmesh[:,0]

    Parameters
    ----------
    filename : string
        Location of the eqdsk output file.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    Rmesh : np.array
        R position.
    Zmesh : np.array
        Z position.
    eqdsk_psi : np.array
        Flux surfaces.
    """
    
    #Open the file
    eqDict=read_eqdsk(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=eqDict['R'][0]
    
    #z-array for equilibrium B
    eqdsk_z=eqDict['Z'][:,0]
    
    #Poloidal flux / 2*pi
    eqdsk_psi=eqDict['psi']
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_flux_surfaces.png'
        
        fig=plt.figure(figsize=(30,8))
        ax=fig.add_subplot(111)
        
        #Plot the poloidal flux
        #Plot limits
        psilim=eqDict['psi_lim']
        #Plot levels
        levels=np.linspace(0,psilim,40)
        
        pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
        pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
        cbar=fig.colorbar(pltobj, ticks = np.round(np.linspace(0, psilim, 10), 4))
        cbar.formatter.set_powerlimits((0, 0))
        cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')
        
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('R [m]')
        
        ax.grid(True)
        
        if saveplot==True:
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
        plt.show()
    
    return Rmesh,Zmesh,eqdsk_psi

def field_along_flux_surface(filename,fluxValue,makeplot=False,saveplot=False):
    """
    This function returns the position and value of the magnetic field strength
    along a given flux surface.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    fluxValue : float
        Value of the flux through a given surface.
    makeplot : boolean
        Make a plot of the data.
    saveplot : boolean
        Save the plot.

    Returns
    -------
    zPosArr : np.array
        Z position.
    rPosArr : np.array
        R position.
    Bflux : np.array
        Strength of the field along the flux surface.
    """
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Flux surface data
    Rmesh,Zmesh,eqdsk_psi=flux_surfaces(filename)
    
    #Magnetic field data
    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    # =========================================================================
    # Process the data
    # =========================================================================
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    #Points for a given contour
    #Make a contour object
    contourObj=plt.contour(Zmesh,Rmesh,eqdsk_psi,levels=[fluxValue])
    #Get the vertex data
    vertexData=contourObj.collections[-1].get_paths()[0].vertices
    #z positions
    zPosArr=vertexData[::-1,0]
    #r positions
    rPosArr=vertexData[::-1,1]
    plt.close()
    
    #Get the field strength at the given points
    #Initialize array
    Bflux=[]
    for i in range(len(zPosArr)):
        Bflux.append(field_interpolator([zPosArr[i],rPosArr[i]])[0])
    #Convert to numpy
    Bflux=np.array(Bflux)
    
    #Calculate the number of maxima of |B| along the flux surface
    epsilon=0.3
    #1st derivative
    dBdz=np.gradient(Bflux,zPosArr)
    #2nd derivative
    d2Bdz2=np.gradient(dBdz,zPosArr)
    #Number of maxima
    maxima=0
    for val in d2Bdz2:
        if val<epsilon and val>-epsilon:
            maxima+=1
    #Print the result
    print('==================================================================')
    print('The number of maxima along this flux surface is- '+str(maxima))
    print('==================================================================')
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        fig=plt.figure(figsize=(21,12))
        fig.suptitle(r'$\Psi_B$='+str(fluxValue)+' T/m$^2$')
        
        ax=fig.add_subplot(211)
        
        ax.plot(zPosArr,Bflux)
        
        ax.set_ylabel('|B| [T]')
        ax.set_title('Field Strength |B|')
        ax.grid(True)
        plt.setp(ax.get_xticklabels(),visible=False)
        
        ax=fig.add_subplot(212)
        
        ax.plot(zPosArr,dBdz)
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'dB/dz [T/m]')
        ax.set_title('dB/dz')
        ax.grid(True)
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_field_along_flux_surface_Psi_'+str(fluxValue)+'.png'
        
            plt.savefig(plotDest+savename,bbox_inches='tight')
            
        plt.show()
    
    return zPosArr,rPosArr,Bflux

def cyclotron_freq(filename,makeplot=False,saveplot=False):
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #Magnetic field data
    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    # =========================================================================
    # Process the data
    # =========================================================================
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    #Points along z
    zPoints=np.linspace(np.min(Zmesh),np.max(Zmesh),250)
    
    #Bz along z for r=0
    Bz=[]
    for i in range(len(zPoints)):
        Bz.append(field_interpolator([zPoints[i],0])[0])
    #Convert to numpy array
    Bz=np.array(Bz)
    
    #Calculate the various cyclotron frequencies
    #Deuterium
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    d5=[]
    #Tritium
    t1=[]
    t2=[]
    t3=[]
    t4=[]
    t5=[]
    t6=[]
    t7=[]
    
    for i in range(len(Bz)):
        
        #Fundamental D
        omegaD=constants.e*Bz[i]/(2*constants.m_p)
        #Fundamental T
        omegaT=constants.e*Bz[i]/(3*constants.m_p)
        
        #Append to the various arrays
        d1.append(omegaD)
        d2.append(2*omegaD)
        d3.append(3*omegaD)
        d4.append(4*omegaD)
        d5.append(5*omegaD)
        t1.append(omegaT)
        t2.append(2*omegaT)
        t3.append(3*omegaT)
        t4.append(4*omegaT)
        t5.append(5*omegaT)
        t6.append(6*omegaT)
        t7.append(7*omegaT)
        
    #Convert to numpy and go to linear frequency in MHz
    d1=np.array(d1)/(1e6*2*np.pi)
    d2=np.array(d2)/(1e6*2*np.pi)
    d3=np.array(d3)/(1e6*2*np.pi)
    d4=np.array(d4)/(1e6*2*np.pi)
    d5=np.array(d5)/(1e6*2*np.pi)
    t1=np.array(t1)/(1e6*2*np.pi)
    t2=np.array(t2)/(1e6*2*np.pi)
    t3=np.array(t3)/(1e6*2*np.pi)
    t4=np.array(t4)/(1e6*2*np.pi)
    t5=np.array(t5)/(1e6*2*np.pi)
    t6=np.array(t6)/(1e6*2*np.pi)
    t7=np.array(t7)/(1e6*2*np.pi)
    
    #Write them in a dictionary to make the return a lot neater
    dFreq={1:d1,
           2:d2,
           3:d3,
           4:d4,
           5:d5,}
    tFreq={1:t1,
           2:t2,
           3:t3,
           4:t4,
           5:t5,
           6:t6,
           7:t7}
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    if makeplot==True:
        
        fig=plt.figure(figsize=(21,12))
        ax=fig.add_subplot(111)
        
        #Plot the deuterium results
        ax.plot(zPoints,d1,color='black',label='Deuterium')
        ax.plot(zPoints,d2,color='black')
        ax.plot(zPoints,d3,color='black')
        ax.plot(zPoints,d4,color='black')
        ax.plot(zPoints,d5,color='black')
        
        #Plot the tritium results
        ax.plot(zPoints,t1,color='blue',linestyle='dashed',label='Tritium')
        ax.plot(zPoints,t2,color='blue',linestyle='dashed')
        ax.plot(zPoints,t3,color='blue',linestyle='dashed')
        ax.plot(zPoints,t4,color='blue',linestyle='dashed')
        ax.plot(zPoints,t5,color='blue',linestyle='dashed')
        ax.plot(zPoints,t6,color='blue',linestyle='dashed')
        ax.plot(zPoints,t7,color='blue',linestyle='dashed')
        
        #Plot the antenna frequencies
        #Deuterium frequency
        freqD=52.5
        ax.plot([np.min(zPoints),np.max(zPoints)],[freqD,freqD],color='black',linestyle='dashed',linewidth=5,label='Antenna Frequency (D)')
        #Tritium frequency
        freqT=35.0
        ax.plot([np.min(zPoints),np.max(zPoints)],[freqT,freqT],color='blue',linestyle='dashed',linewidth=5,label='Antenna Frequency (T)')
        
        #Plot D injection
        dInjCent=-0.63
        ax.plot([dInjCent,dInjCent],[0,1e8],color='red',linewidth=3,label='D (NBI)')
        
        #Plot T injection
        tInjCent=0.63
        ax.plot([tInjCent,tInjCent],[0,1e8],color='red',linestyle='dashed',linewidth=3,label='T (NBI)')
        
        ax.set_xlabel('Z [m]')
        ax.set_ylabel(r'Frequency [MHz]')
        ax.set_title('Cyclotron Frequency Harmonics')
        
        ax.set_xlim(np.min(zPoints),np.max(zPoints))
        ax.set_ylim(0,100)
        ax.legend(loc=[1.01,0.7])
        ax.grid(True)
        
        plt.show()
        
        if saveplot==True:
            
            #Generate the savename of the plot
            #Get the name of the .nc file
            ncName=filename.split('/')[-1]
            #Remove the .nc part
            ncName=ncName[0:-3]
            savename=ncName+'_cyclotron_frequencies.png'
        
            plt.savefig(plotDest+savename,bbox_inches='tight')
    
    return zPoints,Bz,dFreq,tFreq

def plot_field_along_flux_surfaces(filename,fluxValues,saveplot=False):
    """
    This function plots the magnetic field strength and its 1st derivative wrt
    z for a given set of flux values.

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    fluxValues : np.array
        Array with the desired flux values.
    ssaveplot : boolean
        Save the plot.

    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Initialize the plot
    # =========================================================================
    
    fig=plt.figure(figsize=(21,12))
    
    # =========================================================================
    # Plot the field strength
    # =========================================================================
    
    ax=fig.add_subplot(211)
    
    #Number of flux values
    numRays=len(fluxValues)
    #Evenly spaced array for changing colors
    colorArr=np.linspace(0,1,numRays)
    #Go over each flux value
    for i in range(len(fluxValues)):
        
        #Get the data
        zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,fluxValues[i])
        
        #Plot the data
        ax.plot(zPosArr,Bflux,color=(colorArr[i],0,0),label=r'$\Psi_B$='+str(round(fluxValues[i],3)))
    
    ax.set_ylabel('|B| [T]')
    ax.set_title('Field strength |B|')
    
    ax.set_xlim(min(zPosArr),max(zPosArr))
    
    ax.grid(True)
    
    plt.setp(ax.get_xticklabels(),visible=False)
    
    ax.legend(bbox_to_anchor=(1.01,1.03))
    
    # =========================================================================
    # Plot the 1st derivative wrt z
    # =========================================================================
    
    ax=fig.add_subplot(212)
    
    #Go over each flux value
    for i in range(len(fluxValues)):
        
        #Get the data
        zPosArr,rPosArr,Bflux=field_along_flux_surface(filename,fluxValues[i])
        
        #Take the 1st derivative
        dBdz=np.gradient(Bflux,zPosArr)
        
        #Plot the data
        ax.plot(zPosArr,dBdz,color=(colorArr[i],0,0),label=r'$\Psi_B$='+str(round(fluxValues[i],3)))
        
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('dB/dz [T/m]')
    ax.set_title('dB/dz')
    
    ax.set_xlim(min(zPosArr),max(zPosArr))
    
    ax.grid(True)
    
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_field_along_flux_surfaces.png'
        
        fig.savefig(plotDest+savename,bbox_inches='tight')
    
    fig.show()
    
    return

def plot_Bmag_along_r(filename,saveplot=False):
    """
    This function plots the magnitude of the megnetic field as a function of r
    for z=0 (Machine Midplane)

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    ssaveplot : boolean
        Save the plot.

    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the field values
    # =========================================================================

    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    #r array
    rArr=Rmesh[0]
    
    #Bz along z=0
    bMagRadial=Bmag[round(len(Rmesh[:,0])/2)]
    
    #Intialize the plot
    fig=plt.figure(figsize=(12,8))
    ax=plt.subplot(111)
    
    #Plot the data
    ax.plot(rArr,bMagRadial)
    
    #Add labels
    ax.set_xlabel('R [m]')
    ax.set_ylabel('|B| [T]')
    ax.set_title('Field strength along r for z=0')
    
    #Ancilliary
    ax.grid(True)
    
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_field_along_z0.png'
        
        fig.savefig(plotDest+savename,bbox_inches='tight')
        
    #Show the plot
    fig.show()
    
    return

def plot_Bmag_along_z(filename,saveplot=False):
    """
    This function plots the magnitude of the megnetic field as a function of z
    for r=0 (Machine Centerline)

    Parameters
    ----------
    filename : string
        Location of the Genray output file.
    ssaveplot : boolean
        Save the plot.

    Returns
    -------
    None.
    """
    
    # =========================================================================
    # Get the field values
    # =========================================================================

    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    #z array
    zArr=Zmesh[:,0]
    
    #Interpolation function for magnetic field data
    field_interpolator=RegularGridInterpolator((Zmesh[:,0],Rmesh[0]),Bmag)
    
    #Points along z
    zPoints=np.linspace(np.min(Zmesh),np.max(Zmesh),len(zArr))
    
    #Bz along z for r=0
    Bz=[]
    for i in range(len(zPoints)):
        Bz.append(field_interpolator([zPoints[i],0])[0])
    #Convert to numpy array
    Bz=np.array(Bz)
    
    #Intialize the plot
    fig=plt.figure(figsize=(12,8), tight_layout='True')
    ax=plt.subplot(111)
    
    #Plot the data
    ax.plot(zArr,Bz, linewidth=5, zorder=10)
    ax.axhline(np.min(Bz), linewidth=5, color='black', label=r'B$_0$')
    ax.axhline(2*np.min(Bz), linewidth=5, color='black', linestyle='dashed', label=r'2B$_0$')
    
    #Add labels
    ax.legend()
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('|B| [T]')
    ax.set_title('Field strength along z for r=0')
    
    #Ancilliary
    ax.grid(True)
    
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_field_along_r0.png'
        
        fig.savefig(plotDest+savename,bbox_inches='tight')
        
    #Show the plot
    fig.show()
    
    return zArr, Bz

def plot_flux_surfaces_with_resonances(filename,freq,saveplot=False):
    
    #Open the file
    eqDict=read_eqdsk(filename)
    
    # =========================================================================
    # Get the raw data
    # =========================================================================
    
    #r-array for equilibrium B
    eqdsk_r=eqDict['R'][0]
    
    #z-array for equilibrium B
    eqdsk_z=eqDict['Z'][:,0]
    
    #Poloidal flux / 2*pi
    eqdsk_psi=eqDict['psi']
    
    #Magnetic field data
    Rmesh,Zmesh,Br,Bz,Bmag=magnetic_field_RZ(filename)
    
    #Resonance field values
    BDArr,BTArr=resonant_field_strengths(freq)
    
    # =========================================================================
    # Process the data
    # =========================================================================

    #Create the R,Z mesh (useful for plotting)
    Rmesh,Zmesh=np.meshgrid(eqdsk_r,eqdsk_z)
    
    # =========================================================================
    # Plot the data
    # =========================================================================
    
    fig=plt.figure(figsize=(21,8))
    ax=fig.add_subplot(111)
    
    #Plot the poloidal flux
    #Plot limits
    psilim=eqDict['psi_lim']
    #Plot levels
    levels=np.linspace(0,psilim,40)
    
    pltobj=ax.contour(Zmesh,Rmesh,eqdsk_psi,levels=levels)
    pltobj=ax.contour(Zmesh,-Rmesh,eqdsk_psi,levels=levels)
    # cbar=fig.colorbar(pltobj)
    # cbar.set_label(r'$\Psi_B$ [$T \cdot m^2$]')
    
    #Plot the D resonances
    dCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
    ax.contour(Zmesh,-Rmesh,Bmag,levels=BDArr,colors='black',linewidths=3)
    #Plot the T resonances
    tCont=ax.contour(Zmesh,Rmesh,Bmag,levels=BTArr,colors='dodgerblue',linestyles='dashed',linewidths=3,zorder=10)
    ax.contour(Zmesh,-Rmesh,Bmag,levels=BTArr,colors='dodgerblue',linestyles='dashed',linewidths=3,zorder=10)
    
    #Add labels for each contour
    DArrDict=dict()
    TArrDict=dict()
    for i in range(len(BTArr)):
        DArrDict[BDArr[i]]=str(int(BDArr[-1]/BDArr[i]))
        TArrDict[BTArr[i]]=str(int(BTArr[-1]/BTArr[i]))
    
    DLabelLocs=np.array([(-0.75,0.17),(-0.63,0.23),(-0.50,0.23),(-0.28,0.27)])
    TLabelLocs=np.array([(-0.79,0.09),(-0.68,0.12),(-0.59,0.13),(-0.5,0.15)])

    ax.clabel(dCont,fmt=DArrDict,manual=DLabelLocs)
    ax.clabel(tCont,fmt=TArrDict,manual=TLabelLocs)
    
    #Legend
    d1,_=dCont.legend_elements()
    t1,_=tCont.legend_elements()
    ax.legend([d1[0],t1[0]],['Deuterium','Tritium'],loc=[1.01,0.85])
    
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('R [m]')
    
    ax.grid(True)
    
    if saveplot==True:
        
        #Generate the savename of the plot
        #Get the name of the .nc file
        ncName=filename.split('/')[-1]
        #Remove the .nc part
        ncName=ncName[0:-3]
        savename=ncName+'_flux_surfaces_with_resonances.svg'
        
        plt.savefig(plotDest+savename,bbox_inches='tight')
        
    plt.show()
    
    return

if __name__ == '__main__':

    # eqdsk filename from the terminal
    args = parseArgs()
    filenameEqdsk = args.filename

    # Field values
    Rmesh, Zmesh, Br, Bz, Bmag = magnetic_field_RZ(filenameEqdsk)
    
    # Flux values
    _, _, magneticFlux = flux_surfaces(filenameEqdsk)

    # eqdsk dictionary
    eqDict = read_eqdsk(filenameEqdsk)

    # Save them in a .npz file
    np.savez('filenameEqdsk'+'.npz',
             Rmesh = Rmesh,
             Zmesh = Zmesh,
             Br = Br,
             Bz = Bz,
             Bmag = Bmag,
             magneticFlux = magneticFlux,
             psilim = eqDict['psi_lim'],
             eqDict = eqDict)