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

f, ax = plt.subplots(nrows = 5, ncols=6)
for i, a in enumerate(ax.flat):
    a.set_yticks([])
    a.set_xticks([])
    if i >= 5*6-4:
        a.axis('off')

for a, numKmeansSID in zip(ax.flat,range(4,30)):

    # #where to find scene confusion matricies
    # conf_matx_scene_dir = config['supporting directories']['conf_matx_scene']
    # conf_matx_scene_dir = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, conf_matx_scene_dir, numKmeansSID)
    # with h5py.File(conf_matx_scene_dir, 'r') as hf_confmatx:
    #     confmatx_keys = np.array(list(hf_confmatx.keys()))
    #     time_stamps   = [x[-12:] for x in confmatx_keys]
    #     masks         = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_mask']
    #     tables        = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_table']
    #
    #     #only take scenes with >=90% intersect with MAIA L2 grid
    #     #then organize by cloud fraction
    #     #record timestamps to use in accuracy graphs
    #     scenes_gt_90_percent_intersect_L2_grid = []
    #     scenes_by_CF = {'20':[], '40':[], '60':[], '80':[], '100':[]}
    #     for time_stamp, table in zip(time_stamps, table):
    #         table_scne_x = hf_confmatx[table][()]
    #
    #         L2_grid_size = 400*300
    #         num_pixels_in_scene = table.sum()
    #         if num_pixels_in_scene / L2_grid_size >= 0.9:
    #             scenes_gt_90_percent_intersect_L2_grid.append(time_stamp)
    #
    #             #get cloud fraction CF
    #             CF = table[:2].sum()/num_pixels_in_scene
    #             for CF_key in scenes_by_CF:
    #                 if CF <= int(CF_key):
    #                     scenes_by_CF[CF_key].append(time_stamp)

    #now get accuracy by DOY
    scene_accur_home = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, config['supporting directories']['scene_accuracy'], numKmeansSID)
    scene_accur_path = scene_accur_home + '/' + 'scene_ID_accuracy.h5'

    scene_accurs = np.zeros((400,300,46))
    scene_accurs[scene_accurs == 0] = np.nan

    plt.rcParams['font.size'] = 14

    with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
        DOY_bins = list(hf_scene_accur.keys())
        for i, DOY_bin in enumerate(DOY_bins):
            data = hf_scene_accur[DOY_bin+'/MCM_accuracy'][()]
            scene_accurs[:,:,i] = data

    scene_accurs[scene_accurs < 0] = np.nan
    scene_accurs *= 100
    im=a.imshow(np.nanmean(scene_accurs, axis=2), vmin=0, vmax=100)
    # # plt.hist(scene_accurs.flatten(), bins=20)

    scene_accurs = np.nanmean(scene_accurs.flatten())
    a.set_title('SID {:02d};{:2.2f}%'.format(numKmeansSID, scene_accurs))
    print(scene_accurs)
    SID_accur.append(scene_accurs)
    print('SID: ',numKmeansSID)

cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = f.colorbar(im, cax=cb_ax)

plt.figure(2)
x_axis = np.arange(4,30)
plt.plot(x_axis, SID_accur, c='r')
plt.scatter(x_axis, SID_accur, c='blue')
plt.xticks(x_axis, x_axis)
plt.title('Kmeans SID # vs Composite Accuracy\nYears 2004/2010/2018')
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
