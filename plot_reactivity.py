import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# Use TkAgg for interactive plotting
plt.switch_backend('TkAgg')

# Change the font size
plt.rcParams.update({'font.size': 26})

dataObj = np.load('/home/sanwalka/synthetic_proton_detector/reactivity/predicted_reactivity_2d_faster_interpolator.npz')
rArr2D = dataObj['rArr2D']
zArr2D = dataObj['zArr2D']
reactivity2D = dataObj['reactivity2D']

# Total reaction rate for the whole plasma
# Integrate over the whole plasma volume. We can do this by integrating over the 2D grid and multiplying by the volume element (which is 2*pi*r*dr*dz in cylindrical coordinates)
dr = rArr2D[1,0] - rArr2D[0,0]
dz = zArr2D[0,1] - zArr2D[0,0]
print(f'dr: {dr:.3f} m, dz: {dz:.3f} m')

volumeElement = 2*np.pi*rArr2D*dr*dz
totalReactionRate = np.sum(reactivity2D*volumeElement)
print(f'Total reaction rate in the plasma: {totalReactionRate:.2e} #/s')

# Interpolation function for the reactivity
points = np.vstack((rArr2D.flatten(), zArr2D.flatten())).T
reactivityFlat = reactivity2D.flatten()
reactivity_interp_func = sc.interpolate.LinearNDInterpolator(points, reactivityFlat, fill_value=0)

# Put the data onto a uniform grid for plotting
rArr = np.linspace(np.min(rArr2D), np.max(rArr2D), 200)
zArr = np.linspace(np.min(zArr2D), np.max(zArr2D), 200)
RARR, ZARR = np.meshgrid(rArr, zArr)
reactivityUniform = reactivity_interp_func(RARR.flatten(), ZARR.flatten()).reshape(RARR.shape)

fig = plt.figure(figsize=(15, 6), tight_layout=True)
ax = fig.add_subplot(111)

# We are using the inferno colormap
cmap = plt.cm.inferno.copy()
cmap.set_bad(cmap(0))
cmap.set_under(cmap(0))

pltObj = ax.contourf(ZARR, RARR, reactivityUniform, levels=100, cmap='inferno')
# Add the other 4 quadrants by symmetry
ax.contourf(-ZARR, RARR, reactivityUniform, levels=100, cmap='inferno')
ax.contourf(ZARR, -RARR, reactivityUniform, levels=100, cmap='inferno')
ax.contourf(-ZARR, -RARR, reactivityUniform, levels=100, cmap='inferno')

# Also add the limiter flux surface
ax.plot(zArr2D[-1], rArr2D[-1], color='white', label=r'$\Psi_{lim}$', linewidth=4)
ax.plot(-zArr2D[-1], rArr2D[-1], color='white', linewidth=4)
ax.plot(zArr2D[-1], -rArr2D[-1], color='white', linewidth=4)
ax.plot(-zArr2D[-1], -rArr2D[-1], color='white', linewidth=4)

# ax.set_aspect('equal')
ax.set_xlabel('Z [m]')
ax.set_ylabel('R[m]')

ax.legend(loc='upper right')

# Order of magnitude of the reactivity
ordReactivity = int(np.log10(np.max(reactivityUniform)))

# Integer coefficient of the max reactivity
coeffReactivity = int(np.round(np.max(reactivityUniform)/10**ordReactivity))

cbar = fig.colorbar(pltObj, label=r'Reactivity [#/(m$^3$s)]')#, ticks=np.linspace(0, coeffReactivity*(10**ordReactivity), coeffReactivity+1))

plt.savefig('/home/sanwalka/synthetic_proton_detector/plots/predicted_reactivity_2d_with_maxwellian.png', dpi=300)

plt.show()