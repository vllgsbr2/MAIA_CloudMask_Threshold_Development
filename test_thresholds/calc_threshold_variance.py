'''
Author: Javier Villegas Bravo
Purpose: find the variance of the threshold along the OLPs for
         each obs. This will show if thresholds are being
         duplicated across bins and can therefore motivate less
         OLP bins. If the variance is to high, more OLP bins
         should be added.
'''

import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
import h5py
import os

path = '/data/gdi/c/gzhao1/MCM-thresholds/PTAs/LosAngeles/thresh_dev/thresholds/'
thresh_files = [path+x for x in np.sort(os.listdir(path)) if x[0]=='t']
num_SID = 20
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


            thresh_temp = hf_thresh[obs_path][()]
            valid_thresh[:,:,:,:,obs_x,DOY] = hf_thresh[obs_path][()]

#cycle through the NDSI for snow

# thresh_NDSI = valid_thresh[:,:,:,14,1,:]
obs = ['Cirrus', 'NDSI', 'NDVI', 'NIR', 'SVI', 'Vis', 'WI']
results = []
for obs_x in range(7):
    obs_x=1
    #Cirrus
    if obs_x==0:
        thresh_obs_x = valid_thresh[:,:,:,:,obs_x,:]
    #NDSI
    elif obs_x==1:
        thresh_obs_x = valid_thresh[:,:,:,19,obs_x,:]
    #NDVI
    elif obs_x==2:
        thresh_obs_x = valid_thresh[:,:,:,:19,obs_x,:]
    #NIR
    elif obs_x==3:
        thresh_obs_x = valid_thresh[:,:,:,17,obs_x,:]
    #SVI
    elif obs_x==4:
        thresh_obs_x = valid_thresh[:,:,:,:,obs_x,:]
    #Vis
    elif obs_x==5:
        thresh_obs_x = valid_thresh[:,:,:,:17,obs_x,:]
    #WI
    else:
        thresh_obs_x = valid_thresh[:,:,:,:18,obs_x,:]

    with open('./{}_KS_test'.format(obs[obs_x]), 'w') as txt_KS_test:
        for i in range(num_DOY):
            if i==num_DOY-1:
                break
            for j in range(i+1, num_DOY):
                thresh_temp_i = thresh_obs_x[:,:,:,i].flatten()
                thresh_temp_i = thresh_temp_i[thresh_temp_i != -999]

                thresh_temp_j = thresh_obs_x[:,:,:,j].flatten()
                thresh_temp_j = thresh_temp_j[thresh_temp_j != -999]

                if thresh_temp_i.size==0:
                    continue
                if thresh_temp_j.size==0:
                    continue

                KS_test = ks_2samp(thresh_temp_i, thresh_temp_j)

                diff = 'approve null'
                if KS_test[1] < 0.05:
                    diff = 'reject null'

                result = 'KS Test {:1.5f} p-val {:1.5f} DOY {:02d} & {:02d} diff {}'.format(KS_test[0], KS_test[1], i, j, diff)
                results.append(result)
                print(result)
    break

    # thresh_temp = hf_thresh[obs_path][()]
    # thresh_temp[thresh_temp == -999] = np.nan
    # valid_thresh[:,:,:,:,obs_x,DOY] = hf_thresh[obs_path][()]#thresh_temp

    # print(obs_path)

#get variance along each axis independently for each obs
#must be independent by DOY because SID bins 0-10 change meaning throughout
# variance_by_DOY = np.zeros(num_DOY)
# NDSI = 1
# Cirrus = 0
# snow = 14
# NDSI_thresh = valid_thresh[:,:,:,:,Cirrus,:].reshape(num_cosSZA*num_VZA*num_RAZ*num_SID, num_DOY)
# num_bins = 20
# range_ = (0.4,1.)
# hists = np.zeros((num_bins,num_DOY))
# bin_edges = np.zeros((num_bins+1,num_DOY))
# for i in range(num_DOY):
#     data = NDSI_thresh[:,i][NDSI_thresh[:,i] != -999]
#     hists[:,i], bin_edges[:,i] = np.histogram(data, bins=num_bins)#, range=range_)

# plt.figure(1)
# for i in range(num_DOY):
#     # x = hists[:,i]
#     x = NDSI_thresh[:,i][NDSI_thresh[:,i] != -999].flatten()
#     plt.hist(x, bins=num_bins, alpha=0.5)
# plt.show()

# for i in range(num_DOY):
#     for j in range(i+1,num_DOY):
#         diff = bin_NDSI_thresh[:,i] - bin_NDSI_thresh[:,j]
#         plt.plot(diff)
