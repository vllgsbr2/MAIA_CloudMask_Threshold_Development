import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

path = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/thresh_dev/thresholds/'
thresh_files = [path+x for x in np.sort(os.listdir(path)) if x[0]=='t']

for thresh in thresh_files:
    with h5py.File(thresh,'r')as hf_thresh:
        TA = list(hf_thresh.keys())
        DOY = list(hf_thresh[TA[0]].keys())
        obs = list(hf_thresh[TA[0] + '/' + DOY[0]].keys())
        for ob in obs:
            obs_path = '{}/{}/{}'.format(TA[0],DOY[0],ob)
            thresh_temp = hf_thresh[obs_path][()]
            num_thresh = thresh_temp[thresh_temp != -999].shape
            print(obs_path, num_thresh)
