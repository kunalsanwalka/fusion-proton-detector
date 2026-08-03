import os
import sys

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import *

def define_maxwellian(vel, xsi, E, n):
    """
    Create a maxwellian distribution function for a given temperature and density.
    """

    ionMass = 2 * sc.constants.m_p # deuterium mass [kg]
    tempInJoules = E * sc.constants.e # convert from eV to J

    Y, X = np.meshgrid(vel, xsi)

    fMaxwell = np.exp(-ionMass*Y**2/(2*tempInJoules))

    # Normalize the Maxwellian distribution function to n
    integrand = 2 * np.pi * fMaxwell * (Y**2) # spherical coordinates jacobian
    maxwellDens = np.trapezoid(np.trapezoid(integrand, xsi, axis=0), vel, axis=0) * 2
    fMaxwell *= n/maxwellDens

    return fMaxwell

# Dummy arrays for testing the reactivity calculator
zArr2D = np.array([[0.0, 0.0], [0.0, 0.0]])
rArr2D = np.array([[0.0, 0.0], [0.0, 0.0]])

# Define the velocity and pitch-angle grids
eLim = 50 * 1e3 # eV
vel = np.linspace(0, np.sqrt(2*eLim*sc.constants.e/(2*sc.constants.m_p)), 50) # m/s
xsi = np.linspace(0, 1, 50)

Y, X = np.meshgrid(vel, xsi)

# Create a Maxwellian distribution function with a temperature of 10 keV and a density of 1e19 m^-3
E = 10 * 1e3 # eV
n = 1e19 # m^-3
fMaxwell = define_maxwellian(vel, xsi, E, n)

# Tile fMaxwell to match the shape of zArr2D and rArr2D
fMaxwell = np.tile(fMaxwell, (zArr2D.shape[0], zArr2D.shape[1], 1, 1))

print(fMaxwell.shape)

# Calculate the fusion reactivity for the Maxwellian distribution function
zArr2D, rArr2D, reactivity2D = fusion_reactivity_rz(Y, X, zArr2D, rArr2D, fMaxwell)

print(reactivity2D)