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
num_samples_list=[]

with h5py.File(filepath, 'r') as hf:
    bins = list(hf.keys())

    for j in range(20):
        accuracy =[]
        num_samples=[]
        for i, bin_ID in enumerate(bins):
            # print(bin_ID)
            if int(bin_ID[-9:-7]) == j :
                accuracy.append(hf[bin_ID+'/accuracy'][()])
                # print(hf[bin_ID+'/accuracy'][()])
                num_samples.append(hf[bin_ID+'/num_samples'][()])
        s_temp = np.array(accuracy)
        num_samples_temp = np.array(num_samples)
        # s_temp = s_temp[s_temp>=0]
        s_list.append(np.nanmean(s_temp)*100)
        num_samples_list.append(np.nansum(np.array(num_samples)))
# print(s_list)
x = np.arange(20)
# plt.scatter(x, s_list)
# plt.plot(x, s_list, label='accuracy')
# plt.scatter(x, num_samples_list)
# plt.plot(x, num_samples_list)
# plt.xticks(x)#, rotation=295)
# plt.ylim([85,100])
# plt.xlabel('cos(SZA)')
# plt.ylabel('% Accuracy')
#
# plt.show()

#now plot the histogram
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.rcParams['font.size'] = 16

fig, ax = plt.subplots()

fig.suptitle('Accuracy by Surface Type LA PTA')

x_ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'coast','water', 'glint','snow']

color = 'tab:pink'
ax.set_xlabel('Surface Type')
ax.set_ylabel('% Accuracy', color=color)
ax.set_ylim(75, 100)
ax.set_xticks(x)
ax.set_xticklabels(x_ticks)
ax.scatter(x, s_list, color=color)
ax.plot(x, s_list, color=color)
ax.tick_params(axis='y', labelcolor=color)


ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:cyan'
ax1.set_ylabel('number of samples', color=color)  # we already handled the x-label with ax1
ax1.semilogy(x, num_samples_list, color=color)
ax1.set_ylim(0, 10**12)
ax1.tick_params(axis='y', labelcolor=color)

plt.show()
