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

scene_accur_home = PTA_path + '/' + config['supporting directories']['scene_accuracy']
scene_accur_path = scene_accur_home + '/' + 'scene_ID_accuracy.h5'

scene_accurs = np.zeros((400,300,46))
scene_num_samples = np.zeros((400,300,46))
weighted_scene_accurs_by_season = np.zeros((400,300,4))

with h5py.File(scene_accur_path, 'r') as hf_scene_accur:
    DOY_bins = list(hf_scene_accur.keys())
    for i, DOY_bin in enumerate(DOY_bins):
        scene_accurs[:,:,i] = hf_scene_accur[DOY_bin+'/MCM_accuracy'][()]
        scene_num_samples[:,:,i] = np.nansum(hf_scene_accur[DOY_bin+'/num_samples'][()], axis=2)
scene_accurs[scene_accurs == -999] = np.nan
scene_num_samples[scene_num_samples == -999] = np.nan
weighted_scene_accurs = np.nansum(scene_accurs*scene_num_samples, axis=2)/np.nansum(scene_num_samples, axis=2)*100

#divide by season
# DJF/MAM/JJA/SON
# 344-56/64-152/160-240/248-336
# {'DJF':[344/8-1,56/8-1], 'MAM':[64/8-1,152/8-1], 'JJA':[160/8-1,240/8-1], 'SON':[248/8-1,336/8-1]}
# {'DJF': [42.0, 6.0], 'MAM': [7.0, 18.0], 'JJA': [19.0, 29.0], 'SON': [30.0, 41.0]}

# DOY_to_season_dict = {'DJF': [42, 6], 'MAM': [7, 18], 'JJA': [19, 29], 'SON': [30, 41]}
# DOY_to_season_dict = {'DJF': [0, 11], 'MAM': [11, 23], 'JJA': [23, 34], 'SON': [34, 45]}
#
# #roll the two arrays so they start at DJF and end with SON
# #this is so the data can be crunched w/continuous slicing
# scene_accurs_rolled      = np.roll(scene_accurs, 4)
# scene_num_samples_rolled = np.roll(scene_num_samples, 4)
#
# for i, key in enumerate(DOY_to_season_dict):
#     start, end = DOY_to_season_dict[key]
#     weighted_scene_accurs_by_season[:,:,i] = np.nansum(scene_accurs_rolled[:,:,start:end]*\
#                                              scene_num_samples_rolled[:,:,start:end], axis=2)/\
#                                              np.nansum(scene_num_samples_rolled[:,:,start:end], axis=2)*100
#
# f, ax = plt.subplots(nrows=2,ncols=2)
# for i, a in enumerate(ax.flat):
#     a.imshow(weighted_scene_accurs_by_season[:,:,i], vmin=0,vmax=100,cmap=cm.get_cmap('plasma', 20))
#     a.set_xticks([])
#     a.set_yticks([])
#     a.set_title(list(DOY_to_season_dict.keys())[i])

# f, ax = plt.subplots(nrows=6, ncols=8)
# for i, a in enumerate(ax.flat):
#     if i<=45:
#         a.imshow(scene_accurs[:,:,i]*100, vmin=0,vmax=100,cmap=cm.get_cmap('plasma', 20))
#         de = (i+1)*8
#         ds = de-7
#         a.set_title('{:03d}-{:03d}'.format(ds, de))
#     a.set_xticks([])
#     a.set_yticks([])



composit_accuracy = weighted_scene_accurs#np.mean(scene_accurs, axis=2)
plt.imshow(composit_accuracy, vmin=0,vmax=100,cmap=cm.get_cmap('plasma', 100))
plt.xticks([])
plt.yticks([])
plt.title('Composite Accuracy LA PTA 2002-2019')
plt.colorbar()#ticks=np.arange(0,105,5))
plt.rcParams['font.size'] = 18


plt.show()






























# plt.style.use('dark_background')
# fig, ax=plt.subplots(figsize=(10,10))
# cmap = cm.get_cmap('plasma', 20)
# container = []

# #get sfc IDs
# from netCDF4 import Dataset
# filepath_SID = PTA_path + '/' + config['supporting directories']['Surface_IDs']
# filepath_SID = [filepath_SID +'/'+ x for x in os.listdir(filepath_SID) if x[0]=='s']
# SID = np.zeros((400,300,46))
# for i in range(46):
#     with Dataset(filepath_SID[i],'r') as nc_sfcID:
#         SID[:,:,i] = nc_sfcID.variables['surface_ID'][:,:]

#         image = ax.imshow(scene_accurs[:,:,i], cmap=cmap, vmin=0, vmax=100)
#         DOY = (i + 1)*8
#         title = ax.text(0.5,1.05,'Accuracy DOY {:03d}/365\nValid previous 8 days'.format(DOY),
#                         size=plt.rcParams["axes.titlesize"],
#                         ha="center", transform=ax.transAxes, )
#
#         container.append([image, title])
#
# ax.set_yticks([])
# ax.set_xticks([])
# image.cmap.set_under('k')
# #cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
# cbar = plt.colorbar(image,fraction=0.046, pad=0.04)
# cbar.set_ticks(np.arange(0,105,5))
# cbar.set_ticklabels([str(x) for x in np.arange(0,105,5)])
#
# ani = animation.ArtistAnimation(fig, container, interval=700, blit=False,
#                                 repeat=True)
# ani.save('./MCM_Scene_Accuracy_all_DOY_new_grid.mp4')
# s_list = []
# for i in range(46):
#     s_temp = np.copy(scene_accurs[:,:,i])
#     s_temp = s_temp[(s_temp>=0) & (SID[:,:,i] ==12)]
#     s_list.append(np.mean(s_temp))
#
# plt.scatter(np.arange(8,376,8), s_list)
# plt.plot(np.arange(8,376,8), s_list)
# plt.xticks(np.arange(8,376,8), rotation=295)
# plt.ylim([85,100])
# plt.xlabel('Julian Day of Year')
# plt.ylabel('% Accuracy')
