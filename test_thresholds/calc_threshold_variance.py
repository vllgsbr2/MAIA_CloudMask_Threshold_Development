'''
Author: Javier Villegas Bravo
Purpose: find the variance of the threshold along the OLPs for
         each obs. This will show if thresholds are being
         duplicated across bins and can therefore motivate less
         OLP bins. If the variance is to high, more OLP bins
         should be added.
'''

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

path = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/thresh_dev/thresholds/'
thresh_files = [path+x for x in np.sort(os.listdir(path)) if x[0]=='t']
num_SID = 15
num_obs = 7
num_DOY = 46
num_VZA = 15
num_RAZ = 12
num_cosSZA = 10
valid_thresh = np.zeros((num_cosSZA, num_VZA, num_RAZ, num_SID, num_obs, num_DOY))

#collect all thresholds in memory
for DOY, thresh in enumerate(thresh_files):
    with h5py.File(thresh,'r')as hf_thresh:
        TA = list(hf_thresh.keys())
        DOY_ = list(hf_thresh[TA[0]].keys())
        obs = list(hf_thresh[TA[0] + '/' + DOY_[0]].keys())
        for obs_x, ob in enumerate(obs):
            obs_path = '{}/{}/{}'.format(TA[0],DOY_[0],ob)
            # thresh_temp = hf_thresh[obs_path][()]
            # thresh_temp[thresh_temp == -999] = np.nan
            valid_thresh[:,:,:,:,obs_x,DOY] = hf_thresh[obs_path][()]#thresh_temp

            # print(obs_path)

#get variance along each axis independently for each obs
#must be independent by DOY because SID bins 0-10 change meaning throughout
variance_by_DOY = np.zeros(num_DOY)
NDSI = 1
snow = 14
NDSI_thresh = valid_thresh[:,:,:,snow,NDSI,:].reshape(num_cosSZA*num_VZA*num_RAZ, num_DOY)
num_bins = 10
range_ = (0.4,1.)
hists = np.zeros((num_bins,num_DOY))
bin_edges = np.zeros((num_bins+1,num_DOY))
for i in range(num_DOY):
    data = NDSI_thresh[:,i][NDSI_thresh[:,i] != -999]
    hists[:,i], bin_edges[:,i] = np.histogram(data, bins=num_bins, range=range_)

plt.figure(1)
for i in range(num_DOY):
    x = hists[:,i]
    plt.bar(bin_edges[:-1,i], x, alpha=0.5, width=0.5)
plt.show()

# for i in range(num_DOY):
#     for j in range(i+1,num_DOY):
#         diff = bin_NDSI_thresh[:,i] - bin_NDSI_thresh[:,j]
#         plt.plot(diff)
