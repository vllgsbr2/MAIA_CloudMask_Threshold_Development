import numpy as np
import matplotlib.pyplot as plt
import h5py
from rgb_enhancment import get_enhanced_RGB #just takes RGB

def get_R(radiance, SZA, d, E_std_0b):
    """
    convert radiances to Bi-Directional Reflectance Factor, BRF, referred to as 'R'
    [Section 3.3.1.2]
    Convert spectral band radiances to BRF, based on the given solar zenith angle,
    earth sun distance, and the band weighted solar irradiance.
    Arguments:
        radiance {2D narray} -- contains MAIA radiance at any band
        SZA {2D narray} -- same shape as radiance; contains solar zenith angles in degrees
        d {float} -- earth sun distance in Astonomical Units(AU)
        E_std_0b {float} -- band weight solar irradiance at 1 AU,
                            corresponding to the radiance band
    Returns:
        2D narray -- BRF; same shape as radiance
    """
    #now filter out where cosSZA is too small with fill value
    invalid_cosSZA_idx = np.where(np.cos(np.deg2rad(SZA)) <= 0.01)
    radiance[invalid_cosSZA_idx] = -998

    #condition to not step on fill values when converting to BRF(R)
    valid_rad_idx = np.where(radiance >= 0.0)
    radiance[valid_rad_idx] = ((np.pi * radiance * d**2)\
                          / (np.cos(np.deg2rad(SZA)) * E_std_0b))[valid_rad_idx]
    #just assign R to the memory of radiance to highlight conversion
    R = radiance
    return R

def get_timestamps():
    import os
    #choose PTA from keeling
    PTA_file_path   = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data'

    #grab files names for PTA
    filename_MOD_02 = np.array(os.listdir(PTA_file_path + '/MOD_02'))
    filename_MOD_03 = np.array(os.listdir(PTA_file_path + '/MOD_03'))
    filename_MOD_35 = np.array(os.listdir(PTA_file_path + '/MOD_35'))

    #sort files by time so we can access corresponding files without
    #searching in for loop
    filename_MOD_02 = np.sort(filename_MOD_02)
    filename_MOD_03 = np.sort(filename_MOD_03)
    filename_MOD_35 = np.sort(filename_MOD_35)

    #grab time stamp (YYYYDDD.HHMM) to name each group after the granule
    #it comes from
    filename_MOD_02_timeStamp = [x[10:22] for x in filename_MOD_02]

    return filename_MOD_02_timeStamp

home = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data/'


for i in range(1,9):
    #read in file

    filename = home + 'LA_PTA_database_0{}.hdf5'.format(i)
    database = h5py.File(filename, 'r')
    timestamps = list(database.keys())
    #timestamps =  get_timestamps()
    for time in timestamps[::20]:
        #print(time)
        #read in data

        #radiance
        b1_rad = np.array(database[time + '/radiance/band_1'][()]).astype(np.float64) #red
        b3_rad = np.array(database[time + '/radiance/band_3'][()]).astype(np.float64) #blue
        b4_rad = np.array(database[time + '/radiance/band_4'][()]).astype(np.float64) #green

        b2_rad  = np.array(database[time + '/radiance/band_2'][()]).astype(np.float64)  #NIR/NDVI
        b12_rad = np.array(database[time + '/radiance/band_12'][()]).astype(np.float64) #NDSI
        b26_rad = np.array(database[time + '/radiance/band_26'][()]).astype(np.float64) #water vapor

        b1_rad[b1_rad==-999] = np.nan
        b3_rad[b3_rad==-999] = np.nan
        b4_rad[b4_rad==-999] = np.nan
        b2_rad[b2_rad==-999] = np.nan
        b12_rad[b12_rad==-999] = np.nan
        b26_rad[b26_rad==-999] = np.nan

        cloud_mask = np.array(database[time + '/cloud_mask/Unobstructed_FOV_Quality_Flag'][()]).astype(np.float64)
        sun_glint  = np.array(database[time + '/cloud_mask/Sun_glint_Flag'][()]).astype(np.float64)
        Snow_Ice   = np.array(database[time + '/cloud_mask/Snow_Ice_Background_Flag'][()]).astype(np.float64)
        Land_Water = np.array(database[time + '/cloud_mask/Land_Water_Flag'][()]).astype(np.float64)

        #just set not confiedent cloud to clear
        cloud_mask[cloud_mask != 0] = 1

        cloud_mask[cloud_mask==-999] = np.nan
        sun_glint[sun_glint==-999] = np.nan
        Snow_Ice[Snow_Ice==-999] = np.nan
        Land_Water[Land_Water==-999] = np.nan

        #cloud mask tests
        Cloud_Flag_Spatial_Variability = np.array(database[time + '/cloud_mask_tests/Cloud_Flag_Spatial_Variability'][()]).astype(np.float64)
        Cloud_Flag_Visible_Ratio       = np.array(database[time + '/cloud_mask_tests/Cloud_Flag_Visible_Ratio'][()]).astype(np.float64)
        Cloud_Flag_Visible_Reflectance = np.array(database[time + '/cloud_mask_tests/Cloud_Flag_Visible_Reflectance'][()]).astype(np.float64)
        High_Cloud_Flag_1380nm         = np.array(database[time + '/cloud_mask_tests/High_Cloud_Flag_1380nm'][()]).astype(np.float64)
        Near_IR_Reflectance            = np.array(database[time + '/cloud_mask_tests/Near_IR_Reflectance'][()]).astype(np.float64)

        Cloud_Flag_Spatial_Variability[Cloud_Flag_Spatial_Variability==-999] = np.nan
        Cloud_Flag_Visible_Ratio[Cloud_Flag_Visible_Ratio==-999] = np.nan
        Cloud_Flag_Visible_Reflectance[Cloud_Flag_Visible_Reflectance==-999] = np.nan
        High_Cloud_Flag_1380nm[High_Cloud_Flag_1380nm==-999] = np.nan
        Near_IR_Reflectance[Near_IR_Reflectance==-999] = np.nan

        #geolocation
        lat = np.array(database[time + '/geolocation/lat'][()]).astype(np.float64)
        lon = np.array(database[time + '/geolocation/lon'][()]).astype(np.float64)

        lat[lat==-999] = np.nan
        lon[lon==-999] = np.nan

        #sunview geometry
        SZA = np.array(database[time + '/sunView_geometry/solarZenith'][()]).astype(np.float64)
        VZA = np.array(database[time + '/sunView_geometry/sensorZenith'][()]).astype(np.float64)
        VAA = np.array(database[time + '/sunView_geometry/sensorAzimuth'][()]).astype(np.float64)
        SAA = np.array(database[time + '/sunView_geometry/solarAzimuth'][()]).astype(np.float64)

        SZA[SZA==-999] = np.nan
        VZA[VZA==-999] = np.nan
        SAA[SAA==-999] = np.nan
        VAA[VAA==-999] = np.nan

        RAZ = VAA - SAA

        #plot all in one panel
        f, axes = plt.subplots(figsize=(10,10), nrows=4, ncols=4, gridspec_kw = {'wspace':0.0, 'hspace':0.0})

        #stack data
        data = np.dstack((b1_rad, b3_rad, b4_rad, lat,\
                          b2_rad, b12_rad, b26_rad,lon,\
                          cloud_mask, sun_glint, SZA, VZA,
                          Snow_Ice, Land_Water, RAZ, Cloud_Flag_Visible_Reflectance))

        data_names = ['b1_rad', 'b3_rad', 'b4_rad', 'lat',\
                      'b2_rad', 'b12_rad', 'b26_rad','lon',\
                      'cloud_mask', 'sun_glint', 'SZA', 'VZA',
                      'Snow_Ice', 'Land_Water', 'RAZ', 'Visible_Ref\nMOD35']
        cmaps = ['bone', 'bone', 'bone', 'jet',\
                 'bone', 'bone', 'bone','jet',\
                 'binary', 'binary', 'jet', 'jet',
                 'binary', 'binary', 'jet', 'binary']

        #text poition and font
        left, width = .25, .5
        bottom, height = .25, .5
        right = left + width
        top = bottom + height

        font = {'color':  'w',
                'weight': 'heavy',
                'size': 10,
                'backgroundcolor': 'k'
                }

        for i, ax in enumerate(axes.flat):
            #ax.set_aspect('equal')
            ax.imshow(data[:,:,i], cmap=cmaps[i])
            ax.set_ylabel(data_names[i], fontdict=font, rotation='horizontal')
            ax.set_yticks([])
            ax.set_xticks([])
            #ax.text(right, top, data_names[i],\
            #        transform=ax.transAxes, fontdict=font)


        f.suptitle(time, fontdict=font)
        plt.show()
        #f.savefig('/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data/panel_plots/'+time+'.png', dpi=300)

