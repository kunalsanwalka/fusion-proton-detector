# -*- coding: utf-8 -*-
"""
Compares CQL3D's native (r, x, y) distribution function against the
resampled (r, vel, xsi) grid produced by load_cql3d_f, at a chosen flux
surface's midplane.

load_cql3d_f interpolates CQL3D's native grid onto a coarser, uniform
(vel, xsi) grid before the fusion reactivity calculation uses it. Since
that calculation is dominated by whatever slice of velocity space the
fusion cross section weights most heavily (the high-energy tail, where the
cross section is rising steeply), this script exists to check whether that
resampling step is distorting the tail relative to what CQL3D actually
computed - something density-based checks (which are dominated by the
bulk, not the tail) won't catch.

Each `# %%` block below is a runnable cell (VSCode/Spyder/Jupyter).
"""

import os
import sys

import numpy as np
import scipy.constants as constants
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted_fusion_reactivity import load_cql3d_f

# %% Config

filenameCQL3D = '/mnt/n/whamdata/sanwalka/ips_runs/findGasBoxDensity/nneut_1e15_gb_1e18_NBI_800kW_ECH_0kW/simulation_results/9.000/components/fp__cql3dm_4/WHAM.nc'

# Flux surface (midplane R index) to inspect
rIdx = 0

# xsi values to slice along (1 = co-passing/parallel, -1 = counter-passing)
xsiTargets = [1.0, -1.0]

# Fusion-relevant energy band to highlight [keV]
eBandKeV = (5, 25)

# x-axis range for the plot [keV]
ePlotMaxKeV = 40

# y-axis floor for the normalized plot. The fusion-relevant tail typically
# sits many orders of magnitude below the bulk population's peak (that's
# exactly why the reactivity calculation is so sensitive to how well it's
# resolved), so this needs to go much lower than a "reasonable-looking"
# floor like 1e-6 or the tail won't be visible at all.
fPlotFloor = 1e-20

# %% Load native CQL3D f (no resampling)

ds = xr.open_dataset(filenameCQL3D, decode_timedelta=False)

# f dims: (gen_species_dim, rdim, xdimf, ydimf). Species 0 = ions.
fNative_all = ds['f'].values[0]

# x (normalized momentum-per-mass) and vnorm (cm/s) give the native
# velocity grid; this is shared across all flux surfaces.
x = ds['x'].values
vnorm = float(ds['vnorm'].values)
velNative = vnorm * x / 100  # cm/s -> m/s

# y (pitch angle, [0, pi]) is stored per flux surface, so use the row for
# the flux surface actually being inspected here - load_cql3d_f always
# uses row 0 regardless of r, which is a further approximation on top of
# the resampling this script is checking for, so this uses the true
# per-surface grid for a cleaner "native" reference.
y_r = ds['y'].values[rIdx]
xsiNative = 1 - (2 * y_r / np.pi)

fNative_r = np.nan_to_num(fNative_all[rIdx], nan=0.0, posinf=0.0, neginf=0.0)
fNative_r[fNative_r <= 0] = 1e-5

energyNative_keV = 0.5 * (2 * constants.m_p) * velNative**2 / constants.e / 1e3

# %% Load the resampled f produced by load_cql3d_f (what the reactivity
# calculation actually uses)

Y, X, rArr, fResampled = load_cql3d_f(filenameCQL3D)
velResampled = Y[0, :]
xsiResampled = X[:, 0]

energyResampled_keV = 0.5 * (2 * constants.m_p) * velResampled**2 / constants.e / 1e3

# %% Plot f(v) at fixed pitch angle, native vs. resampled, for each xsi target

fig = plt.figure(figsize=(14, 6 * len(xsiTargets)), tight_layout=True)

for panelIdx, xsiTarget in enumerate(xsiTargets):

    ax = fig.add_subplot(len(xsiTargets), 1, panelIdx + 1)

    nativeIdx = np.argmin(np.abs(xsiNative - xsiTarget))
    resampledIdx = np.argmin(np.abs(xsiResampled - xsiTarget))

    fNative_slice = fNative_r[:, nativeIdx]
    fResampled_slice = fResampled[rIdx, resampledIdx, :]

    # Native f is in CQL3D's raw internal units (unnormalized, ~1e20 scale
    # here); resampled f has already been rescaled by load_cql3d_f to match
    # the physical SI density (~1e3 scale). These two are not on comparable
    # absolute scales, so each curve is normalized to its OWN peak - this
    # plot is only meaningful as a SHAPE comparison, not an amplitude one.
    ax.plot(energyNative_keV, fNative_slice / np.max(fNative_slice),
            'o-', markersize=4, label=f'Native CQL3D f ($\\xi$={xsiNative[nativeIdx]:.2f})')
    ax.plot(energyResampled_keV, fResampled_slice / np.max(fResampled_slice),
            's--', markersize=5, label=f'Resampled f ($\\xi$={xsiResampled[resampledIdx]:.2f})')

    ax.axvspan(eBandKeV[0], eBandKeV[1], color='gray', alpha=0.2,
               label=f'{eBandKeV[0]}-{eBandKeV[1]} keV (fusion-relevant)')

    nNativeInBand = np.sum((energyNative_keV >= eBandKeV[0]) & (energyNative_keV <= eBandKeV[1]))
    nResampledInBand = np.sum((energyResampled_keV >= eBandKeV[0]) & (energyResampled_keV <= eBandKeV[1]))
    print(f'xsi~{xsiTarget}: {nNativeInBand} native grid points vs '
          f'{nResampledInBand} resampled grid points in the {eBandKeV} keV band')

    ax.set_yscale('log')
    ax.set_xlim(0, ePlotMaxKeV)
    ax.set_ylim(bottom=fPlotFloor)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel(r'f / max(f) (normalized)')
    ax.set_title(f'r = {rArr[rIdx]:.4f} m, target $\\xi$ = {xsiTarget}')
    ax.legend()

plt.show()
