import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
import configparser
import h5py
import sys

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]

SID_accur = []

#dictionary to store SID accur for each CF
SID_accur_by_CF = {'20':[], '40':[], '60':[], '80':[], '100':[]}
#confusion matrix table for each num SID and each CF
conf_matx_master = np.zeros((26,6,4))
SID_accur_by_CF_AVG_ALL_CF = np.zeros((26))

for numKmeansSID in range(4,30):

    for j, CF_key in enumerate(SID_accur_by_CF):
        #now get accuracy by DOY
        scene_accur_home = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, config['supporting directories']['scene_accuracy'], numKmeansSID)
        scene_accur_path = '{}/scene_ID_accuracy_CF_{:02d}_{:02d}_percent.h5'.format(scene_accur_home, int(CF_key)-20, int(CF_key))

        with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
            DOY_bins = list(hf_scene_accur.keys())
            for DOY_bin in DOY_bins:
                #[true_cloud, true_clear, false_clear, false_cloud]
                conf_matx_master[numKmeansSID-4, :] += hf_scene_accur[DOY_bin+'/total_conf_matx'][()]

        conf_matx_temp = conf_matx_master[numKmeansSID-4, j, :]
        SID_accur_temp = np.sum(conf_matx_temp[:2]) / conf_matx_temp.sum()
        SID_accur_by_CF[CF_key].append(SID_accur_temp)
        print('SID: ',numKmeansSID, SID_accur_temp)

    conf_matx_temp = conf_matx_master[numKmeansSID-4, :, :]
    SID_accur_temp = conf_matx_temp[:, :2].sum() / conf_matx_temp.sum()
    SID_accur_by_CF_AVG_ALL_CF[numKmeansSID-4] = SID_accur_temp * 100

    print(SID_accur_by_CF_AVG_ALL_CF[numKmeansSID-4])


plt.figure(2)
plt.rcParams['font.size'] = 16
x_axis = np.arange(4,30)
colors = ['red', 'yellow', 'green', 'blue', 'purple']

for i, CF_key in enumerate(SID_accur_by_CF):
    label='CF {:02d} - {:02d} %'.format(int(CF_key)-20, int(CF_key))
    data = SID_accur_by_CF[CF_key] * 100
    plt.plot(x_axis   , data, c=colors[i], label=label)
    plt.scatter(x_axis, data, c='black')

    #plot average of all CFs for each num SID
    if i==4:
        label='All CF'
        plt.plot(x_axis   , SID_accur_by_CF_AVG_ALL_CF, c='brown', label=label)
        plt.scatter(x_axis, SID_accur_by_CF_AVG_ALL_CF, c='black')

plt.xticks(x_axis, x_axis)
plt.title('Kmeans SID # vs Composite Accuracy by CF\nYears 2002-2019')
plt.grid()
plt.xlabel('num Kmeans SID (not including snow/water/glint/coast)')
plt.ylabel('% accuracy')
plt.legend()
plt.show()
