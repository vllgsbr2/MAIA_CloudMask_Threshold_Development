from JPL_MCM_threshold_testing import MCM_wrapper
from MCM_output import make_output
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
# test_data_JPL_path = home + 'JPL_data_all_timestamps/test_JPL_data_2002051.1820.HDF5'
test_data_JPL_path = home + 'JPL_data_all_timestamps/test_JPL_data_2009048.1855.HDF5'
Target_Area_X      = 1
# threshold_filepath = home + 'thresholds_all_DOY/thresholds_DOY_049_to_056_bin_06.hdf5'
threshold_filepath = home + 'thresholds_all_DOY/thresholds_DOY_041_to_048_bin_05.hdf5'
# sfc_ID_filepath    = home + 'LA_surface_types/surfaceID_LA_056.nc'
sfc_ID_filepath    = home + 'LA_surface_types/surfaceID_LA_048.nc'
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
scene_type_identifier = \
             MCM_wrapper(test_data_JPL_path, Target_Area_X, threshold_filepath,\
                             sfc_ID_filepath, config_filepath)
plt.figure(6)
scene_type_identifier[scene_type_identifier>0] = np.nan
plt.imshow(scene_type_identifier)
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
cmap = 'bwr'

vmin = -1.2
vmax = 1.2
l,w, = 20,8

#final cloud mask
#f1 = plt.figure(figsize=(20,10))
f1, ax1 = plt.subplots(ncols=2, figsize=(l,w), sharex=True, sharey=True)

ax1[0].imshow(final_cloud_mask, cmap='Greys')
ax1[0].set_title('final MAIA CLoud Mask')

ax1[0].set_xticks([])
ax1[0].set_yticks([])

from rgb_enhancement import *

RGB = np.flip(BRFs[:,:,:3], 2)
RGB[RGB==-999] = np.nan
#RGB = get_enhanced_RGB(RGB)

ax1[1].imshow(RGB)
ax1[1].set_xticks([])
ax1[1].set_yticks([])
ax1[1].set_title('BRF RGB')

#observables
f0, ax0 = plt.subplots(ncols=4, nrows=2, figsize=(l,w),sharex=True, sharey=True)
cmap='bone'
im0 = ax0[0,0].imshow(WI, cmap=cmap+'_r', vmin=0, vmax=WI.max())
im1 = ax0[0,1].imshow(NDVI, cmap=cmap, vmin=-0.6, vmax=0.6)
im2 = ax0[0,2].imshow(NDSI, cmap=cmap, vmin=0.5, vmax=1)
im3 = ax0[0,3].imshow(VIS_Ref, cmap=cmap, vmin=0, vmax=VIS_Ref.max())
im4 = ax0[1,0].imshow(NIR_Ref, cmap=cmap, vmin=0, vmax=NIR_Ref.max())
im5 = ax0[1,1].imshow(SVI, cmap=cmap, vmin=0, vmax=SVI.max())
im6 = ax0[1,2].imshow(Cirrus, cmap=cmap, vmin=0, vmax=1)

ax0[1,3].imshow(RGB)

im0.cmap.set_under('r')
im1.cmap.set_under('r')
im2.cmap.set_under('r')
im3.cmap.set_under('r')
im4.cmap.set_under('r')
im5.cmap.set_under('r')
im6.cmap.set_under('r')

cbar0 = f0.colorbar(im0, ax=ax0[0,0],fraction=0.046, pad=0.04, ticks = np.arange(0,WI.max()+0.2,0.2))
cbar1 = f0.colorbar(im1, ax=ax0[0,1],fraction=0.046, pad=0.04, ticks = np.arange(-1,1.25,0.25))
cbar2 = f0.colorbar(im2, ax=ax0[0,2],fraction=0.046, pad=0.04, ticks = np.arange(-1,1.1,0.1))
cbar3 = f0.colorbar(im3, ax=ax0[0,3],fraction=0.046, pad=0.04, ticks = np.arange(0,VIS_Ref.max()+0.4,0.2))
cbar4 = f0.colorbar(im4, ax=ax0[1,0],fraction=0.046, pad=0.04, ticks = np.arange(0,NIR_Ref.max()+0.1,0.1))
cbar5 = f0.colorbar(im5, ax=ax0[1,1],fraction=0.046, pad=0.04, ticks = np.arange(0,SVI.max()+0.1,0.05))
cbar6 = f0.colorbar(im6, ax=ax0[1,2],fraction=0.046, pad=0.04, ticks = np.arange(0,1.2,0.2))

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



plt.show()
