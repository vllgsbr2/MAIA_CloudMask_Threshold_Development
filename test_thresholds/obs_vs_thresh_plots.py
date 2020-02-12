import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

# home = ''
#
# #Make scatter plot NDVI/NDSI vs whiteness index with a 1 to 1 line
# #testing independence
# obs_list = os.listdir(home + 'observables_60_cores')
# obs_list = [home + x for x in obs_list]
#
# for obs in obs_list:
#     with h5py.File(obs, 'r') as hf_obs:
#         hf_keys = list(hf_obs.keys())
#
#         for time_stamp in hf_keys:
#             WI =

#plot 7 thresholds against sfcID
grouped_list = os.listdir(home + 'test_thresholds/')
#'cosSZA_00_VZA_00_RAZ_00_TA_00_DOY_00_sceneID_00'
#don't count TA and DOY since its the same for all thresholds
n=13
OLP = [[int(group[n+7:n+9])  ,\
        int(group[n+14:n+16]),\
        int(group[n+21:n+23]),\
        int(group[n+45:])] ,\

        for group in grouped_list] #(6, num groups)

grouped_list = [home + x for x in grouped_list]

thresholds = np.zeros((7, len(grouped_list))) #thresh per obs, num total thresh

for i, group in enumerate(grouped_list):
    with h5py.File(group, 'r') as hf_group:
        thresholds[:,i] = hf_group['threhsolds'][()]

hist = np.zeros((7, 50, 15))
hist

f, axes = plt.subplots(ncols=7)
cmap = 'jet'
OLP_labels = ['cos(SZA)', 'VZA', 'RAZ', 'Scene ID']
    for i, ax in enumerate(axes):
        #np.histogram(threshold[i,:], bins=50, range=(0,1))
        ax.scatter(threshold[i,:], OLP[-1,:])
        ax.set_xlabel('{} Thresholds'.format())
        ax.set_ylabel('{}'.format(OLP_labels[-1]))

plt.show()
