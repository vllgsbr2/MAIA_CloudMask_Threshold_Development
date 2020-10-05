import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import configparser
import h5py
# 'scene_Accuracy_DOY_bin_{:02d}'.format(DOY_bin)

config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA           = config['current PTA']['PTA']
PTA_path      = config['PTAs'][PTA]
Target_Area_X = int(config['Target Area Integer'][PTA])

scene_accur_home = PTA_path + '/' + config['supporting directories']['group_accuracy']
scene_accur_path = scene_accur_home + '/' + 'group_ID_accuracy.h5'
filepath = scene_accur_path
plt.rcParams['font.size'] = 16

s_list = []

with h5py.File(filepath, 'r') as hf:
    bins = list(hf.keys())

    for j in range(10):
        accuracy =[]
        num_samples=[]
        for i, bin_ID in enumerate(bins):
            if int(bin_ID[24:26]) == j:
                accuracy.append(hf[bin_ID+'/accuracy'][()])
                num_samples.append(hf[bin_ID+'/num_samples'][()])
        s_temp = np.copy(scene_accurs[:,:,j])
        s_temp = s_temp[s_temp>=0]
        s_list.append(np.mean(s_temp))

x = np.arange(0,1,0.1)
plt.scatter(x, s_list)
plt.plot(x, s_list)
plt.xticks(x)#, rotation=295)
plt.ylim([85,100])
plt.xlabel('cos(SZA)')
plt.ylabel('% Accuracy')

plt.show()
