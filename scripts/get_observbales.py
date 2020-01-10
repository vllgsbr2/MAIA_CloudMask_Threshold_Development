import numpy as np
import matplotlib.pyplot as plt
from svi_dynamic_size_input import svi_calculation
import scipy.misc
import sys
sys.path.insert(0,'/Users/vllgsbr2/Desktop/MODIS_Training/Code')
from plt_MODIS_02 import *
from plt_MODIS_03 import *

#whiteness index
def get_whiteness_index(R_band_6, R_band_5, R_band_4):
    """
    calculate whiteness index

    [Section 3.3.2.1.2]
    whiteness index (WI) uses 3 MAIA spectral bands (4, 5, 6).

    Arguments:
        R_band_6 {2D narray} -- BRF narray for band 6
        R_band_5 {2D narray} -- BRF narray for band 5
        R_band_4 {2D narray} -- BRF narray for band 4

    Returns:
        2D narray -- whiteness index same shape as input arrays
    """

    #data quality house keeping to retain fill values
    whiteness_index = np.ones(np.shape(R_band_6)) * -998
    whiteness_index[(R_band_6 == -999) | (R_band_5 == -999) | (R_band_4 == -999)] = -999
    valid_data_idx = np.where((R_band_6 >= 0) & (R_band_5 >= 0) & (R_band_4 >= 0))

    #calc WI
    visible_average = (R_band_6 + R_band_5 + R_band_4)/3
    whiteness_index[valid_data_idx] = \
            (np.abs(R_band_6 - visible_average)/visible_average + \
             np.abs(R_band_5 - visible_average)/visible_average + \
             np.abs(R_band_4 - visible_average)/visible_average)[valid_data_idx]

    return whiteness_index

#normalized difference vegetation index
def get_NDVI(R_band_6, R_band_9):
    """
    calculate normalized difference vegetation index (NDVI)

    [Section 3.3.2.1.2]
    NDVI uses 2 MAIA spectral bands (6 and 9).

    Arguments:
        R_band_6 {2D narray} -- BRF narray for band 6
        R_band_9 {2D narray} -- BRF narray for band 9

    Returns:
        2D narray -- NDVI same shape as any BRF input
    """

    #data quality house keeping to retain fill values
    NDVI = np.ones(np.shape(R_band_6)) * -998
    NDVI[(R_band_6 == -999) | (R_band_9 == -999)] = -999
    valid_data_idx = np.where((R_band_6 >= 0) & (R_band_9 >= 0))

    NDVI[valid_data_idx] = \
                         ((R_band_9 - R_band_6) / (R_band_9 + R_band_6))[valid_data_idx]

    return NDVI

#normalized difference snow index
def get_NDSI(R_band_5, R_band_12):
    """
    calculate normalized difference snow index (NDVI)

    [Section 3.3.2.1.2]
    NDVI uses 2 MAIA spectral bands (5 and 12).

    Arguments:
        R_band_5 {2D narray} -- BRF narray for band 5
        R_band_12 {2D narray} -- BRF narray for band 12

    Returns:
        2D narray -- NDSI same shape as any BRF input
    """
    #data quality house keeping to retain fill values
    NDSI = np.ones(np.shape(R_band_5)) * -998
    NDSI[(R_band_5 == -999) | (R_band_12 == -999)] = -999
    valid_data_idx = np.where((R_band_5 >= 0) & (R_band_12 >= 0))

    NDSI[valid_data_idx] = \
                         ((R_band_5 - R_band_12) / (R_band_5 + R_band_12))[valid_data_idx]

    return NDSI

#visible reflectance
def get_visible_reflectance(R_band_6):
    """
    return visible BRF of 0.64 um spectral band

    [Section 3.3.2.1.2]
    As the reflectance of band 6 has already been calculated, nothing more will be done.

    Arguments:
        R_band_6 {2D narray} -- BRF narray for band 6

    Returns:
        2D narray -- same as BRF input
    """
    return R_band_6

#near infra-red reflectance
def get_NIR_reflectance(R_band_9):
    """
    return NIR BRF of 0.86 um spectral band

    [Section 3.3.2.1.2]
    As the reflectance of band 9 has already been calculated, nothing more will be done.

    Arguments:
        R_band_9 {2D narray} -- BRF narray for band 9

    Returns:
        2D narray -- same as BRF input
    """
    return R_band_9

#spatial variability index
def get_spatial_variability_index(R_band_6, numrows, numcols):
    """
    calculate spatial variability index (SVI)

    [Section 3.3.2.1.2]
    SVI for a pixel is calculated as the standard deviation of aggregated 1-km R_0.64
    within a 3X3 matrix centered at the pixel.

    Arguments:
        R_band_6 {2D narray} -- BRF narray for band 6

    Returns:
        2D narray -- SVI array with the shape same as BRF input
    """
    import sys
    sys.path.insert(0,'/Users/vllgsbr2/Desktop/MAIA_JPL_code/JPL_MAIA_CM')
    from svi_dynamic_size_input import svi_calculation

    #make copy to not modify original memory
    R_band_6_ = np.copy(R_band_6)
    R_band_6_[R_band_6_ == -998] = -999
    bad_value = -999
    min_valid_pixels = 9
    spatial_variability_index = \
    svi_calculation(R_band_6_, bad_value, min_valid_pixels, numcols, numrows)


    #data quality house keeping
    spatial_variability_index[R_band_6 == -998] = -998
    spatial_variability_index[R_band_6 == -999] = -999

    return spatial_variability_index

#cirrus test
def get_cirrus_Ref(R_band_13):
    """
    return NIR BRF of 1.88 um spectral band

    [Section 3.3.2.1.2]
    As the reflectance of band 13 has already been calculated, nothing more will be done.

    Arguments:
        R_band_13 {2D narray} -- BRF narray for band 13

    Returns:
        2D narray -- same as BRF input
    """
    return R_band_13

#get BRF
def get_BRF_RGB(filename_MOD_02, filename_MOD_03, path):

    #get Ref RGB to compare by eye
    cos_sza = np.cos(np.deg2rad(get_solarZenith(path + filename_MOD_03)))
    fieldnames_list  = ['EV_500_Aggr1km_RefSB', 'EV_250_Aggr1km_RefSB', 'EV_1KM_RefSB']
    rad_or_ref = False #True for radiance, False for reflectance
    #make channels for RGB photo (index 01234 -> band 34567)
    blu = prepare_data(path + filename_MOD_02, fieldnames_list[0],rad_or_ref)[0,:,:] #band 3 from 500 meter res
    grn = prepare_data(path + filename_MOD_02, fieldnames_list[0],rad_or_ref)[1,:,:] #band 4 from 500 meter res
    red = prepare_data(path + filename_MOD_02, fieldnames_list[1],rad_or_ref)[0,:,:] #band 1 from 250 meter res

    NIR_2 = prepare_data(path + filename_MOD_02, fieldnames_list[1],rad_or_ref)[1,:,:] #band 2 from 250 meter res
    NIR_6 = prepare_data(path + filename_MOD_02, fieldnames_list[0],rad_or_ref)[3,:,:] #band 6 from 500 meter res
    NIR_26 = prepare_data(path + filename_MOD_02, fieldnames_list[2],rad_or_ref)[-1,:,:] #band 1 from 250 meter res
    #convert to BRF
    blu /= cos_sza
    grn /= cos_sza
    red /= cos_sza

    NIR_2  /= cos_sza
    NIR_6 /= cos_sza
    NIR_26 /= cos_sza

    #to MAIA spec
    NIR_9 = NIR_2
    NIR_12 = NIR_6
    NIR_13 = NIR_26

    return red, grn, blu, NIR_9, NIR_12, NIR_13
def get_enhanced_RGB(RGB):
    def scale_image(image):
        along_track = image.shape[0]
        cross_track = image.shape[1]

        x = np.array([0,  30,  60, 120, 190, 255], dtype=np.uint8)
        y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)

        scaled = np.zeros((along_track, cross_track), dtype=np.uint8)
        for i in range(len(x)-1):
            x1 = x[i]
            x2 = x[i+1]
            y1 = y[i]
            y2 = y[i+1]
            m = (y2 - y1) / float(x2 - x1)
            b = y2 - (m *x2)
            mask = ((image >= x1) & (image < x2))
            scaled = scaled + mask * np.asarray(m * image + b, dtype=np.uint8)

        mask = image >= x2
        scaled = scaled + (mask * 255)
        return scaled


    # case = ['/home/javi/MODIS_Training/BRF_RGB_Toronto.npz',
    #         '/home/javi/MODIS_Training/BRF_RGB_Aerosol.npz' ]
    # rgb = np.load(case[1])['arr_0'][:]
    enhanced_RGB = np.zeros_like(RGB, dtype=np.uint8)
    for i in range(3):
        enhanced_RGB[:, :, i] = scale_image(scipy.misc.bytescale(RGB[:, :, i]))

    return enhanced_RGB
home = '../LA_PTA_2017_12_16_1805/2017_09_03_1855/'
filename_MOD_02 = home + 'MOD021KM.A2017246.1855.061.2017258202757.hdf'
filename_MOD_03    = home + 'MOD03.A2017246.1855.061.2017257170030.hdf'
# MOD35_path    = home + 'MOD35_L2.A2017246.1855.061.2017258203025.hdf'

R = get_BRF_RGB(filename_MOD_02, filename_MOD_03, '')

#just make a panel plot
R_band_6  = R[0]
R_band_5  = R[1]
R_band_4  = R[2]
R_band_9  = R[3]
R_band_12 = R[4]
R_band_13 = R[5]

numrows, numcols = np.shape(R_band_6)[0], np.shape(R_band_6)[1]

WI = get_whiteness_index(R_band_6, R_band_5, R_band_4)
NDVI = get_NDVI(R_band_6, R_band_9)
NDSI = get_NDSI(R_band_5, R_band_12)
VIS_Ref = get_visible_reflectance(R_band_6)
NIR_Ref = get_NIR_reflectance(R_band_9)
SVI = get_spatial_variability_index(R_band_6, numrows, numcols)
Cirrus = get_cirrus_Ref(R_band_13)
RGB = np.dstack((R_band_6, R_band_5, R_band_4))
RGB_enhanced = get_enhanced_RGB(RGB)

cmap_wi='bone_r'
cmap_ndsi = 'Blues'
cmap_ndvi = 'Greens'
cmap_svi = 'jet'
cmap_nirr = 'bone'
cmap_vr = 'Reds'
cmap_cirrus = 'Blues'


plt.rcParams.update({'font.size': 22})


# plt.figure()
# plt.imshow(RGB_enhanced)
# plt.title('RGB')
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(wi, cmap = cmap_wi)
# plt.title('Whiteness Index')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(ndsi, cmap = cmap_ndsi)
# plt.title('Normalized Difference Snow Index')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(ndvi, cmap = cmap_ndvi)
# plt.title('Normalized Difference Vegetation Index')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(svi, cmap = cmap_svi, vmin=0, vmax=svi.max())
# plt.title('Spatial Variability Index')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(nirr, cmap = cmap_nirr)
# plt.title('NIR Reflectance')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(vr, cmap = cmap_vr)
# plt.title('Red Band Reflectance')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# plt.figure()
# plt.imshow(cirrus, cmap = cmap_cirrus)
# plt.title('Water Vapor Absorbing Channel')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()

#observables
plt.rcParams.update({'font.size': 18})
plt.style.use('dark_background')
cmap = 'bone'
vmin, vmax= -1, 1
l,w = 16, 9
f0, ax0 = plt.subplots(ncols=4, nrows=2, figsize=(l,w))
im = ax0[0,0].imshow(WI, cmap=cmap+'_r', vmin=vmin, vmax=vmax)
ax0[0,1].imshow(NDVI, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[0,2].imshow(NDSI, cmap=cmap+'_r', vmin=vmin, vmax=vmax)
ax0[0,3].imshow(VIS_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,0].imshow(NIR_Ref, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,1].imshow(SVI*10, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,2].imshow(Cirrus*10, cmap=cmap, vmin=vmin, vmax=vmax)
ax0[1,3].imshow(RGB_enhanced, cmap=cmap, vmin=vmin, vmax=vmax)


ax0[0,0].set_title('WI')
ax0[0,1].set_title('NDVI')
ax0[0,2].set_title('NDSI')
ax0[0,3].set_title('VIS_Ref')
ax0[1,0].set_title('NIR_Ref')
ax0[1,1].set_title('SVI')
ax0[1,2].set_title('Cirrus')
ax0[1,3].set_title('RGB')

for ax in ax0.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('./observables.jpg', dpi=300)
#plt.show()
