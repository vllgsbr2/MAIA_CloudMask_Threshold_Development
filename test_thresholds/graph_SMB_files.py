'''
Author     : Javier Villegas Bravo
Association: UIUC and NASA JPL
Date       : 10/15/20

PURPOSE:
graph cross sections of SMB data to find how many k-means clusters we should
choose for the surface type scheme
'''
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os

SMB_home = '/data/gdi/c/gzhao1/MCM-surfaceID/SMB/LosAngeles/'
SMB_files = [SMB_home + x for x in os.listdir(SMB_home) if x[:3]=='max']

SMB_file_x = SMB_files[0]

with Dataset(SMB_file_x, 'r') as nc_SMB:
    #holds max_BRF with dims (x, y, cos_sza, vza, raz) -> (400,300,10,15,12)
    max_BRF = nc_SMB.variables['max_BRF'][:]
    cos_sza = nc_SMB.variables['cos_sza'][:]
    vza = nc_SMB.variables['vza'][:]
    raz = nc_SMB.variables['raz'][:]

max_BRF_flat = max_BRF.reshape((400*300,10,15,12))
max_BRF_flat[max_BRF_flat<0] = np.nan
max_BRF_by_SVC = np.zeros((10,15,12))
for i in range(cos_sza.shape[0]):
    for j in range(vza.shape[0]):
        for k in range(raz.shape[0]):
            max_BRF_by_SVC[i,j,k] = np.nanmean(max_BRF_flat[:,i,j,k])
    print(i)

max_BRF_by_SVC = max_BRF_by_SVC.reshape(10*15*12)
max_BRF_by_SVC = np.sort(max_BRF_by_SVC)

#every time the next SVGC is more than 5% bigger than the last draw a thresh
#first get the gradient
max_BRF_by_SVC_grad = np.gradient(max_BRF_by_SVC)


f, ax = plt.subplots(ncols=1,nrows=1)


ax.bar(np.arange(10*15*12), max_BRF_by_SVC_grad, c='r', label='gradient', alpha=0.3)
ax.set_ylabel('Gradient')
ax.tick_params(axis='y', labelcolor='tab:red')

ax1 = ax.twinx()
ax1.plot(np.arange(10*15*12), max_BRF_by_SVC, c='b', label='BRDF')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xlabel('Sun View Geometry Combinations 0-1799')
ax1.set_ylabel('Mean BRDF for SVGC')

plt.show()
