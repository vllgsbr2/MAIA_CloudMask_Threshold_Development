import numpy as np
import pandas as pd
import h5py
from netCDF4 import Dataset
#JPL inputs**********************************************************************
#land_water and snow_ice mask are not known as of June 2019
def get_JPL_data(test_data_JPL_path):

    JPL_file = h5py.File(test_data_JPL_path, 'r')

    #retrieve radiances
    rad_band_4  = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_4']
    rad_band_5  = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_5']
    rad_band_6  = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_6']
    rad_band_9  = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_9']
    rad_band_12 = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_12']
    rad_band_13 = JPL_file['Anicillary_Radiometric_Product/Radiance/rad_band_13']

    #retrieve radiance quality flag 'RDQI'
    RDQI_b4  = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_4']
    RDQI_b5  = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_5']
    RDQI_b6  = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_6']
    RDQI_b9  = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_9']
    RDQI_b12 = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_12']
    RDQI_b13 = JPL_file['Anicillary_Radiometric_Product/Radiometric_Data_Quality_Indicator/RDQI_band_13']

    #retrieve Solar Zenith, Sensor Zenith, Solar Azimuth, Sensor Azimuth
    #assumed sun-view geometry is in degrees
    #S=sun; V=sensor/viewing; ZA=zenith angle; AA=azimuthal angle
    SZA     = JPL_file['Anicillary_Radiometric_Product/Sun_View_Geometry/solar_zenith_angle']
    VZA     = JPL_file['Anicillary_Radiometric_Product/Sun_View_Geometry/viewing_zenith_angle']
    SAA     = JPL_file['Anicillary_Radiometric_Product/Sun_View_Geometry/solar_azimuth_angle']
    VAA     = JPL_file['Anicillary_Radiometric_Product/Sun_View_Geometry/viewing_azimuth_angle']

    #retrieve current earth sun distance 'd' in AU; is d in AU just E0?
    d = JPL_file['Anicillary_Radiometric_Product/Earth_Sun_Distance/earth_sun_dist_in_AU'][()]

    #retrieve standard band weighted solar irradiance at 1AU 'E_std_0b'
    #[w/m**2/um]
    #in order MAIA  bands 6,9,4,5,12,13
    #in order MODIS bands 1,2,3,4,6 ,26
    E_std_0 = JPL_file['Anicillary_Radiometric_Product/Band_Weighted_Solar_Irradiance_at_1AU/Band_Weighted_Solar_Irradiance_at_1AU']

    #snow/ice mask
    #0 for clear, 1 for snow/ice
    #note it should be downscaled from 4km res to 1km res
    #expected snow/ice mask:
    #Remote Sensing of Environment 196 (2017) 42-55
    #Global Multisensor Automated satellite-based Snow and Ice Mapping System
    #(GMASI) for cryosphere monitoring
    #Peter Romanov
    #NOAA-CREST, City University of New York, 160 Convent Ave, New York, NY,
    #USA Office of Satellite Applications and Research, NOAA NESDIS,
    #5830 University Research Court, College Park, MD 20740, USA
    snow_ice_mask = JPL_file['Anicillary_Geometric_Product/Snow_Ice_Mask/Snow_Ice_Mask']

    DOY = JPL_file.get('Anicillary_Radiometric_Product/Day_of_year/Day_of_year')[()]

    Target_Area = JPL_file.get('Anicillary_Radiometric_Product/Target_Area/Target_Area')[()]

    return rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12, rad_band_13,\
           RDQI_b4, RDQI_b5, RDQI_b6, RDQI_b9, RDQI_b12, RDQI_b13,\
           SZA, VZA, SAA, VAA,\
           d,\
           E_std_0,\
           snow_ice_mask,\
           DOY,\
           Target_Area

#retrieve UIUC ancillary datasets**************************************
#thresholds, surface ID , and configuration file are provided by us

def get_UIUC_data(sfc_ID_filepath, config_filepath, DOY):
    '''
    sfc_ID_filepath {str} -- path to surface ID for a particular PTA and DOY bin
    config_filepath {str} -- path to configuration file
    '''

    #thresholds
    #(SZA, VZA, RAZ, Target Area, land_water, snowice,sfc_ID, DOY, sun_glint)
    #each test has the above array that can be indexed for the threshold

    #land Surface ID
    with h5py.File(sfc_ID_filepath, 'r') as sfc_ID_file:
        DOY_bin = np.digitize(DOY, np.arange(8,376,8), right=True)
        DOY_end = (DOY_bin+1)*8
        DOY_start = DOY_end - 7
        path = '{:03d}/sfcid'.format(DOY_start)
        sfc_ID = sfc_ID_file[path][()]

    #read config file
    config_data = pd.read_csv(config_filepath, skiprows=3, header=0)
    Sun_glint_exclusion_angle  = float(config_data['Sun-glint exclusion angle'])
    Max_RDQI                   = int(config_data['Max RDQI'])
    Max_valid_DTT              = int(config_data['Max valid DTT'])
    Min_valid_DTT              = int(config_data['Min valid DTT'])
    fill_val_1                 = int(config_data['fill val 1'])
    fill_val_2                 = int(config_data['fill val 2'])
    fill_val_3                 = int(config_data['fill val 3'])
    Min_num_of_activated_tests = int(config_data['Min num of activated tests'])
    activation_values          = [float(config_data['activation val 1']),\
                                  float(config_data['activation val 2']),\
                                  float(config_data['activation val 3']),\
                                  float(config_data['activation val 4']),\
                                  float(config_data['activation val 5'])  ]

    return sfc_ID,\
           Sun_glint_exclusion_angle,\
           Max_RDQI,\
           Max_valid_DTT,\
           Min_valid_DTT,\
           fill_val_1,\
           fill_val_2,\
           fill_val_3,\
           Min_num_of_activated_tests,\
           activation_values