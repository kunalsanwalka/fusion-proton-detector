import pickle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})
plt.switch_backend('TkAgg')

# Load the data
dataObj = np.load('/home/sanwalka/synthetic_proton_detector/reactivity/volume_weights_test.npz')
dataObj = np.load('/home/sanwalka/synthetic_proton_detector/reactivity/volume_weights_0.7m_10deg.npz')

# Extract the arrays
detPosX = dataObj['detPosX']
detPosY = dataObj['detPosY']
detZLoc = dataObj['detZLoc']

openingPosX = dataObj['openingPosX']
openingPosY = dataObj['openingPosY']

xPosWeights = dataObj['xPosWeights']
yPosWeights = dataObj['yPosWeights']
weights = dataObj['weights']
distances = dataObj['distance']

# Load the particle tracks
with open('/home/sanwalka/synthetic_proton_detector/particle_tracks/track_data_0.7m_10deg.pkl', 'rb') as f:
    xzTracks = pickle.load(f)

# Set all 0 distances to a large value
distances = np.where(distances == 0, 1e6, distances)

solidAngleNorm = 1/(4*np.pi*distances**2)
solidAngleNorm = np.nan_to_num(solidAngleNorm, nan=0.0, posinf=0.0, neginf=0.0)
weights *= solidAngleNorm

weights /= np.max(weights)

# Make an interpolation function for the weights
points = np.vstack((xPosWeights.flatten(), yPosWeights.flatten())).T
weightsFlat = weights.flatten()
weights_interp_func = sc.interpolate.LinearNDInterpolator(points, weightsFlat, fill_value=0)

# Interpolate the weights onto a regular grid for plotting
xi = np.linspace(np.min(xPosWeights), np.max(xPosWeights), 20)
yi = np.linspace(np.min(yPosWeights), np.max(yPosWeights), 20)
XARR, YARR = np.meshgrid(xi, yi)
weightsUniform = weights_interp_func(XARR.flatten(), YARR.flatten()).reshape(XARR.shape)

weightsUniform = np.nan_to_num(weightsUniform, nan=0.0, posinf=0.0, neginf=0.0)
weightsUniform /= np.max(weightsUniform)
weightsUniform = np.clip(weightsUniform, 0, 1)

# Plot the data
fig = plt.figure(figsize=(9, 9), tight_layout=True)
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')

# Plot the detector and opening positions
ax.plot(detPosX, detPosY, 
        linewidth=6,
        label='Detector',
        color='C0',
        zorder=10)
ax.plot(openingPosX, openingPosY, 
        linewidth=3, 
        label='Collimator',
        color='C1',
        zorder=9)

# Plot 2 circles from the detector to the opening to show the collimating tube
# Use the 2 points and radius formula

circ1Rad = 0.7 + 0.0127
circ2Rad = 0.7 - 0.0127

# Points that define the circles
circ1X1 = detPosX.min()
circ1X2 = openingPosX.min()
circ1Y1 = detPosY.min()
circ1Y2 = openingPosY.min()

circ2X1 = detPosX.max()
circ2X2 = openingPosX.max()
circ2Y1 = detPosY.max()
circ2Y2 = openingPosY.max()

# Distance between the points
dist1 = np.sqrt((circ1X2 - circ1X1)**2 + (circ1Y2 - circ1Y1)**2)
dist2 = np.sqrt((circ2X2 - circ2X1)**2 + (circ2Y2 - circ2Y1)**2)

# Middle point between the two points
midX1 = (circ1X1 + circ1X2)/2
midY1 = (circ1Y1 + circ1Y2)/2
midX2 = (circ2X1 + circ2X2)/2
midY2 = (circ2Y1 + circ2Y2)/2

# Center of the circles
h1 = midX1 + (((circ1Y1-circ1Y2)/dist1) * np.sqrt(circ1Rad**2 - (dist1/2)**2))
h2 = midX2 + (((circ2Y1-circ2Y2)/dist2) * np.sqrt(circ2Rad**2 - (dist2/2)**2))

k1 = midY1 - (((circ1X2-circ1X1)/dist1) * np.sqrt(circ1Rad**2 - (dist1/2)**2))
k2 = midY2 - (((circ2X2-circ2X1)/dist2) * np.sqrt(circ2Rad**2 - (dist2/2)**2))

# x array to plot the circles
xArr1 = np.linspace(circ1X1, circ1X2, 100)
xArr2 = np.linspace(circ2X1, circ2X2, 100)

# y array for the circles
yArr1 = k1 + np.sqrt(circ1Rad**2 - (xArr1 - h1)**2)
yArr2 = k2 + np.sqrt(circ2Rad**2 - (xArr2 - h2)**2)

# Plot the collimating tube circles
ax.plot(xArr1, 2*midY1-yArr1, color='C1', linewidth=3, zorder=9)
ax.plot(xArr2, 2*midY2-yArr2, color='C1', linewidth=3, zorder=9)
        
# Plot a circle marking the plasma boundary
theta = np.linspace(0, 2*np.pi, 100)
r_plasma = 0.06
x_plasma = r_plasma * np.cos(theta)
y_plasma = r_plasma * np.sin(theta)
ax.plot(x_plasma, y_plasma, 
        linewidth=6,
        color='grey',
        label=r'$\Psi_{lim}$',
        zorder=8)

# Plot the weights as a filled contour
pltObj1 = ax.contourf(XARR, YARR, weightsUniform,
                      levels = np.linspace(0, 1, 100),
                      cmap='inferno',
                      zorder=1)

# Plot the particle tracks
for i in range(len(xzTracks)):
     track = xzTracks[i]
     ax.plot(track[0], track[1], 
             color='grey', 
             linewidth=0.3,
             alpha=0.5,
             zorder=7)

ax.legend()

ax.set_xlim(-0.3, 0.1)
ax.set_ylim(-0.1, 0.35)

ax.set_aspect('equal')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'z={np.round(detZLoc, 2)}m')

plt.savefig('/home/sanwalka/synthetic_proton_detector/plots/volume_weights_0.7m_10deg.png', dpi=300)

plt.show()