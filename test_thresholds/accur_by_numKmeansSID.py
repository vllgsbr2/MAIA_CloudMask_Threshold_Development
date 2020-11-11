import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import configparser
import h5py
import sys
# 'scene_Accuracy_DOY_bin_{:02d}'.format(DOY_bin)

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]

SID_accur = []

for numKmeansSID in range(4,30):
    scene_accur_home = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, config['supporting directories']['scene_accuracy'], numKmeansSID)
    scene_accur_path = scene_accur_home + '/' + 'scene_ID_accuracy.h5'

    scene_accurs = np.zeros((400,300,46))

    plt.rcParams['font.size'] = 16

    with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
        DOY_bins = list(hf_scene_accur.keys())
        for i, DOY_bin in enumerate(DOY_bins):
            data = hf_scene_accur[DOY_bin+'/MCM_accuracy'][()]
            scene_accurs[:,:,i] = data*100

    scene_accurs[scene_accurs < 0] = np.nan
    # plt.imshow(scene_accurs[:,:,0])
    # plt.colorbar()
    plt.hist(scene_accurs.flatten(), bins=20)
    plt.show()
    scene_accurs                   = np.nanmean(scene_accurs.flatten())
    print(scene_accurs)
    SID_accur.append(scene_accurs)
    print('SID: ',numKmeansSID)

plt.plot(np.arange(4,30), SID_accur)
plt.title('Kmeans SID # vs Composite Accuracy\nYears 2004/2010/2018')

plt.show()
