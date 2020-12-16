from JPL_MCM_threshold_testing import MCM_wrapper
from MCM_output import make_output
import h5py
import numpy as np
import configparser
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')


config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#LA
# test_data_JPL_path = home + 'JPL_data_all_timestamps/test_JPL_data_2009048.1855.HDF5'
# test_data_JPL_path = home + 'JPL_data_all_timestamps/test_JPL_data_2010197.1845.HDF5'
# test_scene = 'test_JPL_data_2019120.1920.h5' # ok general case
# test_scene = 'test_JPL_data_2018002.1850.h5' # cirrus event
# test_scene = 'test_JPL_data_2019148.1805.h5' # ok general case
# test_scene = 'test_JPL_data_2019139.1815.h5' #good scene just cut off
test_scene = 'test_JPL_data_2019137.1825.h5' # great scene!! would like more water clouds
# test_scene = 'test_JPL_data_2019096.1830.h5' #great scene no water clouds :(
# test_scene = 'test_JPL_data_2018365.1830.h5' #this is it!!

#test some Guangyu scenes
# test_scene = 'test_JPL_data_2019276.1805.h5'
num_Kmeans_sfc_types = 16#11
guangyu_home = '/data/gdi/c/gzhao1/MCM-thresholds/PTAs/LosAngeles/thresh_dev/thresholds/'
# threshold_filepath = guangyu_home + 'thresholds_DOY_273_to_280_bin_34.h5'
# threshold_filepath = guangyu_home + 'OBthresholds_DOY_273_to_280_bin_34.h5'
test_scene = 'test_JPL_data_2019276.1805.h5'
test_scene = 'test_JPL_data_2006029.1910.h5'

# test_scene = 'test_JPL_data_2019089.1825.h5'
DOY = int(test_scene[18:-8])
DOY_bin = np.digitize(DOY, np.arange(8,376,8), right=True)
DOY_end = (DOY_bin+1)*8
DOY_start = DOY_end - 7

# thresh_file = 'thresholds_DOY_{:03d}_to_{:03d}_bin_{:02d}_numSID_{:02d}.h5'.format(DOY_start, DOY_end, DOY_bin, num_Kmeans_sfc_types)
thresh_file = 'thresholds_DOY_{:03d}_to_{:03d}_bin_{:02d}.h5'.format(DOY_start, DOY_end, DOY_bin)
threshold_filepath = guangyu_home + thresh_file
# SID_file    = 'num_Kmeans_SID_{:02d}/surfaceID_LosAngeles_{:03d}.nc'.format(num_Kmeans_sfc_types, DOY_end)
SID_file    = 'surfaceID_LosAngeles_{:03d}.nc'.format(DOY_end)
sfc_ID_filepath = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/Surface_IDs/num_Kmeans_SID_16/' + SID_file

print(test_scene, thresh_file, SID_file)

test_data_JPL_path = '{}/{}/{}'.format(PTA_path, config['supporting directories']['MCM_Input'],test_scene)
# threshold_filepath = '{}/{}/{}'.format(PTA_path, config['supporting directories']['thresh'], thresh_file)
# sfc_ID_filepath    = '{}/{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'], SID_file)
print(threshold_filepath)
# test_data_JPL_path = '{}/{}/{}'.format(PTA_path, config['supporting directories']['MCM_Input'],'test_JPL_data_2003114.1845.h5')
# threshold_filepath = '{}/{}/{}'.format(PTA_path, config['supporting directories']['thresh'], 'thresholds_DOY_113_to_120_bin_14.h5')
# sfc_ID_filepath    = '{}/{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'], 'surfaceID_LosAngeles_120.nc')

# #old stuff
# threshold_filepath = '{}/{}/{}'.format(PTA_path, config['supporting directories']['thresh'], 'old_SID_thresh/thresholds_DOY_113_to_120_bin_14.h5')
# sfc_ID_filepath    = '{}/{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'], 'old_SID/surfaceID_LosAngeles_120.nc')
Target_Area_X      = int(config['Target Area Integer'][PTA])
#ROME
# test_data_JPL_path = '{}/{}/{}'.format(PTA_path, config['supporting directories']['MCM_Input'],'test_JPL_data_2010190.1020.h5')
# threshold_filepath = '{}/{}/{}'.format(PTA_path, config['supporting directories']['thresh'], 'thresholds_DOY_185_to_192_bin_23.h5')
# sfc_ID_filepath    = '{}/{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'], 'surfaceID_Rome_192.nc')
# Target_Area_X      = int(config['Target Area Integer'][PTA])


config_filepath    = './config.csv'

#run MCM
Sun_glint_exclusion_angle,\
Max_RDQI,\
Max_valid_DTT,\
Min_valid_DTT,\
fill_val_1,\
fill_val_2,\
fill_val_3,\
Min_num_of_activated_tests,\
activation_values,\
observable_data,\
DTT, final_cloud_mask,\
BRFs,\
SZA, VZA, VAA,SAA,\
scene_type_identifier,\
T = \
             MCM_wrapper(test_data_JPL_path, Target_Area_X, threshold_filepath,\
                             sfc_ID_filepath, config_filepath, num_Kmeans_sfc_types)
print(scene_type_identifier.max(), scene_type_identifier[scene_type_identifier > -1].min())
#save output
make_output(Sun_glint_exclusion_angle,\
            Max_RDQI,\
            Max_valid_DTT,\
            Min_valid_DTT,\
            fill_val_1,\
            fill_val_2,\
            fill_val_3,\
            Min_num_of_activated_tests,\
            activation_values,\
            observable_data,\
            DTT, final_cloud_mask,\
            BRFs,\
            SZA, VZA, VAA,SAA,\
            scene_type_identifier)

#grab mod35 cm from input file
with h5py.File(test_data_JPL_path, 'r') as hf_output:
    mod35cm = hf_output['MOD35_cloud_mask'][()]

import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
plt.figure()
cmap = ListedColormap(['white', 'green', 'blue','black'])
norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
plt.imshow(mod35cm, cmap=cmap, norm=norm)
cbar = plt.colorbar()
cbar.set_ticks([0.5,1.5,2.5,3.5])
cbar.set_ticklabels(['cloudy', 'uncertain\nclear', \
                     'probably\nclear', 'confident\nclear'])
plt.xticks([])
plt.yticks([])


# plt.show()


#reformat the return for plotting
WI, NDVI, NDSI, VIS_Ref, NIR_Ref, SVI, Cirrus = observable_data[:,:,0],\
                                                observable_data[:,:,1],\
                                                observable_data[:,:,2],\
                                                observable_data[:,:,3],\
                                                observable_data[:,:,4],\
                                                observable_data[:,:,5],\
                                                observable_data[:,:,6]
DTT_WI, DTT_NDVI, DTT_NDSI, DTT_VIS_Ref, DTT_NIR_Ref, DTT_SVI, DTT_Cirrus =\
                                                DTT[:,:,0],\
                                                DTT[:,:,1],\
                                                DTT[:,:,2],\
                                                DTT[:,:,3],\
                                                DTT[:,:,4],\
                                                DTT[:,:,5],\
                                                DTT[:,:,6]
#plotting*******************************************************************
print('Commence Plotting Output')
plt.rcParams['font.size'] = 18
plt.figure(6)

im_scene_ID = plt.imshow(scene_type_identifier, vmin=0)
im_scene_ID.cmap.set_under('pink')
plt.xticks([])
plt.yticks([])

cmap = 'bwr'

vmin = -1.2
vmax = 1.2
l,w, = 20,8

#final cloud mask
#f1 = plt.figure(figsize=(20,10))
f1, ax1 = plt.subplots(ncols=2, figsize=(l,w), sharex=True, sharey=True)

im_cm = ax1[0].imshow(final_cloud_mask, cmap='Greys', vmax=1.01)
ax1[0].set_title('Final MAIA Cloud Mask')

ax1[0].set_xticks([])
ax1[0].set_yticks([])
im_cm.cmap.set_over('r')

from rgb_enhancement import *

RGB = np.flip(BRFs[:,:,:3], 2)
RGB[RGB==-999] = 0
RGB = get_enhanced_RGB(RGB)
# RGB = RGB.astype(dtype=np.float)
# RGB[RGB==0] = np.nan

# RGB = (RGB * 255).astype(np.uint8)
ax1[1].imshow(RGB)
ax1[1].set_xticks([])
ax1[1].set_yticks([])
ax1[1].set_title('BRF RGB')

#observables
f0, ax0 = plt.subplots(ncols=4, nrows=2, figsize=(l,w),sharex=True, sharey=True)
cmap='bone'
im0 = ax0[0,0].imshow(WI     , cmap=cmap+'_r', vmin=0, vmax = 0.6 )
im1 = ax0[0,1].imshow(NDVI   , cmap='PRGn'   , vmin=-0.4, vmax=0.4)
im2 = ax0[0,2].imshow(NDSI   , cmap='PRGn'   , vmin=-0.5 , vmax=0.5)
im3 = ax0[0,3].imshow(VIS_Ref, cmap=cmap     , vmin=0, vmax=0.8   )#, vmax=VIS_Ref.max())
im4 = ax0[1,0].imshow(NIR_Ref, cmap=cmap     , vmin=0, vmax=0.8  )#, vmax=NIR_Ref.max())
im5 = ax0[1,1].imshow(SVI    , cmap=cmap     , vmin=0, vmax=0.25)#, vmax=SVI.max())
im6 = ax0[1,2].imshow(Cirrus , cmap=cmap     , vmin=0, vmax=0.2   )#, vmax=1)

ax0[1,3].imshow(RGB)

im0.cmap.set_under('c')
im1.cmap.set_under('c')
im2.cmap.set_under('c')
im3.cmap.set_under('c')
im4.cmap.set_under('c')
im5.cmap.set_under('c')
im6.cmap.set_under('c')

cbar0 = f0.colorbar(im0, ax=ax0[0,0],fraction=0.046, pad=0.04)#)#, ticks = np.arange(0,WI.max()+0.2,0.2))
cbar1 = f0.colorbar(im1, ax=ax0[0,1],fraction=0.046, pad=0.04)#)#, ticks = np.arange(-1,1.25,0.25))
cbar2 = f0.colorbar(im2, ax=ax0[0,2],fraction=0.046, pad=0.04)#)#, ticks = np.arange(-1,1.1,0.1))
cbar3 = f0.colorbar(im3, ax=ax0[0,3],fraction=0.046, pad=0.04)#)#, ticks = np.arange(0,VIS_Ref.max()+0.4,0.2))
cbar4 = f0.colorbar(im4, ax=ax0[1,0],fraction=0.046, pad=0.04)#)#, ticks = np.arange(0,NIR_Ref.max()+0.1,0.1))
cbar5 = f0.colorbar(im5, ax=ax0[1,1],fraction=0.046, pad=0.04)#)#, ticks = np.arange(0,SVI.max()+0.1,0.05))
cbar6 = f0.colorbar(im6, ax=ax0[1,2],fraction=0.046, pad=0.04)#)#, ticks = np.arange(0,1.2,0.2))

font_size = 10 # Adjust as appropriate.
cbar0.ax.tick_params(labelsize=font_size)
cbar1.ax.tick_params(labelsize=font_size)
cbar2.ax.tick_params(labelsize=font_size)
cbar3.ax.tick_params(labelsize=font_size)
cbar4.ax.tick_params(labelsize=font_size)
cbar5.ax.tick_params(labelsize=font_size)
cbar6.ax.tick_params(labelsize=font_size)

ax0[0,0].set_title('WI')
ax0[0,1].set_title('NDVI')
ax0[0,2].set_title('NDSI')
ax0[0,3].set_title('VIS_Ref')
ax0[1,0].set_title('NIR_Ref')
ax0[1,1].set_title('SVI')
ax0[1,2].set_title('Cirrus')
ax0[1,3].set_title('BRF RGB')

for a in ax0.flat:
    a.set_xticks([])
    a.set_yticks([])
    a.axis('off')


#DTT
vmin = -101
vmax = 101

cmap = cm.get_cmap('bwr')
#cmap.set_bad(color='black')
cmap.set_under('black')

f, ax = plt.subplots(ncols=4, nrows=2, figsize=(l,w), sharex=True, sharey=True)
im = ax[0,0].imshow(DTT_WI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,1].imshow(DTT_NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,2].imshow(DTT_NDSI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,3].imshow(DTT_VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,0].imshow(DTT_NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,1].imshow(DTT_SVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,2].imshow(DTT_Cirrus, cmap=cmap, vmin=vmin, vmax=vmax)

ax[1,3].imshow(RGB)

ax[0,0].set_title('DTT_WI')
ax[0,1].set_title('DTT_NDVI')
ax[0,2].set_title('DTT_NDSI')
ax[0,3].set_title('DTT_VIS_Ref')
ax[1,0].set_title('DTT_NIR_Ref')
ax[1,1].set_title('DTT_SVI')
ax[1,2].set_title('DTT_Cirrus')
ax[1,3].set_title('BRF RGB')

cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = f.colorbar(im, cax=cb_ax)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

#thresholds
#Thresholds
l,w, = 20,8
import matplotlib.pyplot as plt
import matplotlib.cm as cm
f2, ax2 = plt.subplots(ncols=4, nrows=2, figsize=(l,w),sharex=True, sharey=True)
cmap    = cm.get_cmap('jet')
T[T==-999] = np.nan
im0 = ax2[0,0].imshow(T[:,:,0], cmap=cmap)
im1 = ax2[0,1].imshow(T[:,:,1], cmap=cmap, vmax=0.2)
im2 = ax2[0,2].imshow(T[:,:,2], cmap=cmap)
im3 = ax2[0,3].imshow(T[:,:,3], cmap=cmap, vmax=0.5)
im4 = ax2[1,0].imshow(T[:,:,4], cmap=cmap)
im5 = ax2[1,1].imshow(T[:,:,5], cmap=cmap, vmax=0.15)
im6 = ax2[1,2].imshow(T[:,:,6], cmap=cmap, vmax=0.04)
im0.cmap.set_under('k')
im1.cmap.set_under('k')
im2.cmap.set_under('k')
im3.cmap.set_under('k')
im4.cmap.set_under('k')
im5.cmap.set_under('k')
im6.cmap.set_under('k')
ax2[0,0].set_title('WI')
ax2[0,1].set_title('NDVI')
ax2[0,2].set_title('NDSI')
ax2[0,3].set_title('VIS_Ref')
ax2[1,0].set_title('NIR_Ref')
ax2[1,1].set_title('SVI')
ax2[1,2].set_title('Cirrus')
cbar0 = f2.colorbar(im0, ax=ax2[0,0],fraction=0.046, pad=0.04)#, ticks = np.arange(0,WI.max()+0.2,0.2))
cbar1 = f2.colorbar(im1, ax=ax2[0,1],fraction=0.046, pad=0.04)#, ticks = np.arange(-1,1.25,0.25))
cbar2 = f2.colorbar(im2, ax=ax2[0,2],fraction=0.046, pad=0.04)#, ticks = np.arange(-1,1.1,0.1))
cbar3 = f2.colorbar(im3, ax=ax2[0,3],fraction=0.046, pad=0.04)#, ticks = np.arange(0,VIS_Ref.max()+0.4,0.2))
cbar4 = f2.colorbar(im4, ax=ax2[1,0],fraction=0.046, pad=0.04)#, ticks = np.arange(0,NIR_Ref.max()+0.1,0.1))
cbar5 = f2.colorbar(im5, ax=ax2[1,1],fraction=0.046, pad=0.04)#, ticks = np.arange(0,SVI.max()+0.1,0.05))
cbar6 = f2.colorbar(im6, ax=ax2[1,2],fraction=0.046, pad=0.04)#, ticks = np.arange(0,1.2,0.2))

for a in ax2.flat:
    a.set_xticks([])
    a.set_yticks([])

ax2[1,3].axis('off')



plt.show()
