import numpy as np
from pyhdf.SD import SD
import h5py
import matplotlib.pyplot as plt

def make_JPL_data_from_MODIS(database_path, time_stamp):
    
    with h5py.File(database_path, 'r') as hf_database:
        #read MODIS data
        rads_b4      = hf_database['{}/radiance/band_3'.format(time_stamp)][()]
        rads_b5      = hf_database['{}/radiance/band_4'.format(time_stamp)][()]
        rads_b6      = hf_database['{}/radiance/band_1'.format(time_stamp)][()]
        rads_b9      = hf_database['{}/radiance/band_2'.format(time_stamp)][()]
        rads_b12     = hf_database['{}/radiance/band_6'.format(time_stamp)][()]
        rads_b13     = hf_database['{}/radiance/band_26'.format(time_stamp)][()] 

        rad_scales   = hf_database['{}/scale_factors/radiance'.format(time_stamp)][()]
        ref_scales   = hf_database['{}/scale_factors/reflectance'.format(time_stamp)][()]
        E_std_0      = np.pi * rad_scales / ref_scales

        lat          = hf_database['{}/geolocation/lat'.format(time_stamp)][()]
        lon          = hf_database['{}/geolocation/lon'.format(time_stamp)][()]
        SZA          = hf_database['{}/SunView_geometry/solarZenith'.format(time_stamp)][()]
        VZA          = hf_database['{}/SunView_geometry/sensorZenith'.format(time_stamp)][()]
        SAA          = hf_database['{}/SunView_geometry/solarAzimuth'.format(time_stamp)][()]
        VAA          = hf_database['{}/SunView_geometry/sensorAzimuth'.format(time_stamp)][()]
        modcm        = hf_database['{}/cloud_mask/Unobstructed_FOV_Quality_Flag'.format(time_stamp)][()]
        snowice_mask = hf_database['{}/cloud_mask/Snow_Ice_Background_Flag'.format(time_stamp)][()]
        water_mask   = hf_database['{}/cloud_mask/Land_Water_Flag'.format(time_stamp)][()]

    #create hdf5 file
    outpath = 'test_JPL_data_TERRA_MODIS_{}.HDF5'.format(time_stamps)
    hf = h5py.File(outpath, 'w')

    #define arbitrary shape for granule/orbit
    shape = (1000,1000)

    #create structure in hdf file
    ARP                = hf.create_group('Anicillary_Radiometric_Product')
    ARP_rad            = ARP.create_group('Radiance')
    ARP_RDQI           = ARP.create_group('Radiometric_Data_Quality_Indicator')
    ARP_SVG            = ARP.create_group('Sun_View_Geometry')
    ARP_earth_sun_dist = ARP.create_group('Earth_Sun_distance')
    ARP_E_std0b        = ARP.create_group('Band_Weighted_Solar_Irradiance_at_1AU')
    ARP_DOY            = ARP.create_group('Day_of_year')
    ARP_TA             = ARP.create_group('Target_Area')

    AGP     = hf.create_group('Anicillary_Geometric_Product')
    AGP_LWM = AGP.create_group('Land_Water_Mask')
    AGP_SIM = AGP.create_group('Snow_Ice_Mask')

    #convert
    rad_b4  = rads[2]
    rad_b5  = rads[3]
    rad_b6  = rads[0]
    rad_b9  = rads[1]
    rad_b12 = rads[5]
    rad_b13 = rads[7]

    RDQI_b4  = np.zeros(shape)
    RDQI_b5  = np.zeros(shape)
    RDQI_b6  = np.zeros(shape)
    RDQI_b9  = np.zeros(shape)
    RDQI_b12 = np.zeros(shape)
    RDQI_b13 = np.zeros(shape)

    earth_sun_dist = 1

    E_std0b = E_std_0

    DOY = 6

    TA = 0

    #0 for snow_ice, 1 for non
    #should be consistant with MOD35 definition
    snow_ice_mask = snowice_mask


    # np.place(snow_ice_mask, snow_ice_mask==0, 2)
    # np.place(snow_ice_mask, snow_ice_mask==1, 0)
    # np.place(snow_ice_mask, snow_ice_mask==2, 1)

    #0 for water and 1 for land
    water_mask[water_mask > 2] = 1
    water_mask[water_mask !=1] = 0

    land_water_mask = water_mask
    # print(np.where(land_water_mask==0)[1].size, np.where(land_water_mask==1)[1].size)
    # plt.imshow(land_water_mask)
    # plt.colorbar()
    # plt.show()
    # print(land_water_mask)


    #assign data to groups
    ARP_rad.create_dataset('rad_band_4' , data=rad_b4 , dtype='f')
    ARP_rad.create_dataset('rad_band_5' , data=rad_b5 , dtype='f')
    ARP_rad.create_dataset('rad_band_6' , data=rad_b6 , dtype='f')
    ARP_rad.create_dataset('rad_band_9' , data=rad_b9 , dtype='f')
    ARP_rad.create_dataset('rad_band_12', data=rad_b12, dtype='f')
    ARP_rad.create_dataset('rad_band_13', data=rad_b13, dtype='f')

    ARP_RDQI.create_dataset('RDQI_band_4' , data=RDQI_b4 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_5' , data=RDQI_b5 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_6' , data=RDQI_b6 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_9' , data=RDQI_b9 , dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_12', data=RDQI_b12, dtype='i4')
    ARP_RDQI.create_dataset('RDQI_band_13', data=RDQI_b13, dtype='i4')

    ARP_SVG.create_dataset('solar_zenith_angle'   , data=SZA, dtype='f')
    ARP_SVG.create_dataset('viewing_zenith_angle' , data=VZA, dtype='f')
    ARP_SVG.create_dataset('solar_azimuth_angle'  , data=SAA, dtype='f')
    ARP_SVG.create_dataset('viewing_azimuth_angle', data=VAA, dtype='f')

    ARP_earth_sun_dist.create_dataset('earth_sun_dist_in_AU', data=earth_sun_dist)

    ARP_E_std0b.create_dataset('Band_Weighted_Solar_Irradiance_at_1AU', data=E_std0b, dtype='f')

    ARP_DOY.create_dataset('Day_of_year', data=DOY)

    ARP_TA.create_dataset('Target_Area', data=TA)

    AGP_LWM.create_dataset('Land_Water_Mask', data=land_water_mask, dtype='i4', compression='gzip')
    AGP_SIM.create_dataset('Snow_Ice_Mask', data=snow_ice_mask, dtype='i4', compression='gzip')

    hf.close()

    return lat, lon, modcm


