from JPL_MCM import MCM_wrapper
from MCM_output import make_output
import matplotlib.pyplot as plt

test_data_JPL_path = 'test_JPL_MODIS_data_.HDF5'
Target_Area_X      = 1
threshold_filepath = '/Users/vllgsbr2/Desktop/keeling_test_files/thresholds_MCM.hdf5'
sfc_ID_filepath    = '/Users/vllgsbr2/Desktop/SurfaceID_LA_048.nc'
config_filepath    = '/Users/vllgsbr2/Desktop/MAIA_JPL_code/ancillary_UIUC_data/config.csv'

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
f1, ax1 = plt.subplots(ncols=2, figsize=(l,w))#, sharex=)

ax1[0].imshow(final_cloud_mask, cmap='Greys')
#ax1[0].set_title('final MAIA CLoud Mask')

ax1[0].set_xticks([])
ax1[0].set_yticks([])

#import RGB
# from cartopy import config
# import cartopy.crs as ccrs
import sys
sys.path.insert(0,'../../misc_code')
from rgb_enhancement import *
home = '/Users/vllgsbr2/Desktop/LA_PTA_2017_12_16_1805/2017_09_03_1855/'
MOD021KM_path = home + 'MOD021KM.A2017246.1855.061.2017258202757.hdf'
MOD03_path    = home + 'MOD03.A2017246.1855.061.2017257170030.hdf'

filename_MOD_02 = MOD021KM_path
filename_MOD_03 =MOD03_path
path = ''
RGB = get_BRF_RGB(filename_MOD_02, filename_MOD_03, path)
RGB = get_enhanced_RGB(RGB)
i1, i2 = 530, 1530
j1, j2 = 177, 1177
ax1[1].imshow(RGB[i1:i2, j1:j2])
ax1[1].set_xticks([])
ax1[1].set_yticks([])


#observables
f0, ax0 = plt.subplots(ncols=4, nrows=2, figsize=(l,w))
im = ax0[0,0].imshow(WI, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[0,1].imshow(NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[0,2].imshow(NDSI, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[0,3].imshow(VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,0].imshow(NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,1].imshow(SVI*2, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,2].imshow(Cirrus, cmap=cmap, vmin=vmin, vmax=vmax)

ax0[1,3].imshow(RGB[i1:i2, j1:j2])

ax0[0,0].set_title('WI')
ax0[0,1].set_title('NDVI')
ax0[0,2].set_title('NDSI')
ax0[0,3].set_title('VIS_Ref')
ax0[1,0].set_title('NIR_Ref')
ax0[1,1].set_title('SVI x2 scaling')
ax0[1,2].set_title('Cirrus')
ax0[1,3].set_title('BRF RGB')


cb_ax = f0.add_axes([0.93, 0.1, 0.02, 0.8])
cbar = f0.colorbar(im, cax=cb_ax)

for a in ax0.flat:
    a.set_xticks([])
    a.set_yticks([])

#DTT
vmin = -127
vmax = 101

f, ax = plt.subplots(ncols=4, nrows=2, figsize=(l,w))
im = ax[0,0].imshow(DTT_WI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,1].imshow(DTT_NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,2].imshow(DTT_NDSI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[0,3].imshow(DTT_VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,0].imshow(DTT_NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,1].imshow(DTT_SVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax[1,2].imshow(DTT_Cirrus, cmap=cmap, vmin=vmin, vmax=vmax)

ax[1,3].imshow(RGB[i1:i2, j1:j2])

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
