import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
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

# f, ax = plt.subplots(nrows = 5, ncols=6)
# for i, a in enumerate(ax.flat):
#     a.set_yticks([])
#     a.set_xticks([])
#     if i >= 5*6-4:
#         a.axis('off')
#
# # build a rectangle in axes coords
# left, width = 0., .5
# bottom, height = 0., .5
# right = left + width
# top = bottom + height

plt.rcParams['font.size'] = 16

#distionary to store SID accur for each CF
SID_accur_by_CF = {'20':[], '40':[], '60':[], '80':[], '100':[]}

# for a, numKmeansSID in zip(ax.flat,range(4,30)):
for numKmeansSID in range(4,30):
    for CF_key in SID_accur_by_CF:
        #now get accuracy by DOY
        scene_accur_home = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, config['supporting directories']['scene_accuracy'], numKmeansSID)
        # scene_accur_path = scene_accur_home + '/' + 'scene_ID_accuracy.h5'
        scene_accur_path = '{}/scene_ID_accuracy_CF_{:02d}_{:02d}_percent.h5'.format(scene_accur_home, int(CF_key)-20, int(CF_key))

        scene_accurs = np.zeros((400,300,46))
        scene_accurs[scene_accurs == 0] = np.nan

        with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
            DOY_bins = list(hf_scene_accur.keys())
            for i, DOY_bin in enumerate(DOY_bins):
                data = hf_scene_accur[DOY_bin+'/MCM_accuracy'][()]
                scene_accurs[:,:,i] = data

        scene_accurs[scene_accurs < 0] = np.nan
        scene_accurs *= 100
        # im=a.imshow(np.nanmean(scene_accurs, axis=2), vmin=0, vmax=100)
        # # plt.hist(scene_accurs.flatten(), bins=20)

        scene_accurs = np.nanmean(scene_accurs.flatten())
        # label_graph = 'SID {:02d};{:2.2f}%'.format(numKmeansSID, scene_accurs)

        # axes coordinates are 0,0 is bottom left and 1,1 is upper right
        # p = patches.Rectangle(
        #     (left, bottom), width, height,
        #     fill=False, transform=a.transAxes, clip_on=False
        #     )
        # a.text(left, bottom, label_graph,
        #     horizontalalignment='left',
        #     verticalalignment='bottom',
        #     transform=a.transAxes)

        print(scene_accurs)
        SID_accur_by_CF[CF_key].append(scene_accurs)
        # SID_accur.append(scene_accurs)
        print('SID: ',numKmeansSID)


cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = f.colorbar(im, cax=cb_ax)

plt.figure(2)
x_axis = np.arange(4,30)
colors = ['red', 'yellow', 'green', 'blue']
for i, CF_key in enumerate(SID_accur_by_CF):
    label='CF {:02d} - {:02d} %'.format(int(CF_key)-20, int(CF_key))
    plt.plot(x_axis, SID_accur_by_CF[CF_key], c=colors[i], label=label)
    plt.scatter(x_axis, SID_accur_by_CF[CF_key], c='black')

plt.xticks(x_axis, x_axis)
plt.title('Kmeans SID # vs Composite Accuracy by CF\nYears 2002-2019')
plt.grid()
plt.xlabel('num Kmeans SID (not including snow/water/glint/coast)')
plt.ylabel('% accuracy')
plt.show()
#*******************************************************************************





#*******************************************************************************
# scene_accur_home = '{}/{}/'.format(PTA_path, config['supporting directories']['scene_accuracy'])
# scene_accur_path = scene_accur_home + 'scene_ID_accuracy.h5'
#
# scene_accurs = np.zeros((400,300,46))
#
# plt.rcParams['font.size'] = 16
#
# with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
#     DOY_bins = list(hf_scene_accur.keys())
#     for i, DOY_bin in enumerate(DOY_bins):
#         data = hf_scene_accur[DOY_bin+'/MCM_accuracy'][()]
#         scene_accurs[:,:,i] = data
#
# scene_accurs[scene_accurs < 0] = np.nan
# scene_accurs *= 100
# plt.hist(scene_accurs.flatten(), bins=40)
#
# scene_accurs                   = np.nanmean(scene_accurs.flatten())
# print(scene_accurs)
# plt.show()
