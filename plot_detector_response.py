import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
plt.rcParams.update({'font.size': 24})

# Open the file
dataObj = np.load('/home/sanwalka/synthetic_proton_detector/reactivity/detector_response_keisuke.npz')

# Extract the arrays
detResponseArr = dataObj['detResponseArr']
detPosArr = dataObj['detPosArr']

# Plot the detector response as a function of z-position
zPosArr = detPosArr[:, 2]

fig = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig.add_subplot(111)

ax.plot(zPosArr, detResponseArr,  linewidth=3)
ax.scatter(zPosArr, detResponseArr, s=300)

ax.set_ylim(0, None)
ax.set_xlabel('Z [m]')
ax.set_ylabel('Detector Rate [#/ms]')

plt.savefig('/home/sanwalka/synthetic_proton_detector/plots/detector_response_plot_keisuke.png', dpi=300)

plt.show()