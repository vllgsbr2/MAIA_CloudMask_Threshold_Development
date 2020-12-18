import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

path = '/data/gdi/c/gzhao1/MCM-thresholds/PTAs/LosAngeles/thresh_dev/thresholds/'
thresh_files = [path+x for x in np.sort(os.listdir(path)) if x[0]=='t']
num_SID = 15
num_obs = 7
num_DOY = 46
num_sunview = 10*15*12
valid_thresh = np.zeros((num_sunview, num_SID, num_obs, num_DOY))

for DOY, thresh in enumerate(thresh_files):
    with h5py.File(thresh,'r')as hf_thresh:
        TA = list(hf_thresh.keys())
        DOY = list(hf_thresh[TA[0]].keys())
        obs = list(hf_thresh[TA[0] + '/' + DOY[0]].keys())
        for ob in obs:
            obs_path = '{}/{}/{}'.format(TA[0],DOY[0],ob)
            thresh_temp = hf_thresh[obs_path][()]
            num_thresh = thresh_temp[thresh_temp != -999].shape
            thresh_temp[thresh_temp == -999] = np.nan
            thresh_temp = thresh_temp.reshape((num_sunview, num_SID, num_obs, num_DOY))
            valid_NDSI_thresh[:,DOY] = thresh_temp

            # print(obs_path, num_thresh)
