'''
AUTHORS: Javier Villegas Bravo
         Yizhe Zhan,
         Guangyu Zhao,
         University of Illinois at Urbana-Champaign
         Department of Atmospheric Sciences
         Sept 2019
This is the MAIA L2 Cloud Mask algorithm
Reference doc: MAIA Level 2 Cloud Mask Algorithm Theoretical Basis JPL-103722
'''

import numpy as np
import sys
from fetch_MCM_input_data import *
import time
from svi_dynamic_size_input import svi_calculation #svi_calculation import svi_calculation

#quality check radiance*********************************************************
#section 3.2.1.3 and 3.3.2.1.2 (last paragraph)
def mark_bad_radiance(radiance, RDQI, Max_RDQI):
    """
    check radiance quality

    [Sections 3.2.1.3 and 3.3.2.1.2]
    Specify low-quality radiance data (RDQI larger than a prescribed value as
    defined in configuration file) and missing data (RDQI==3).
    These data are set to -998 and -999, respectively.

    Arguments:
        radiance {2D narray} -- contains MAIA radiance at any band
        RDQI {2D narray} -- same shape as radiance; contains radiance quality flag
        Max_RDQI {integer} -- from configuration file;
                              denotes minimum usable quality flag

    Returns:
        2D narray -- radiance array with fill values denoting unusable or missing data
    """

    radiance[(RDQI > Max_RDQI) & (RDQI < 3)] = -998
    #delete meaningless negative radiance without stepping on -999 fill val
    radiance[(radiance < 0) & (radiance > -998)] = -998
    radiance[RDQI == 3] = -999

    return radiance

#calculate ancillary datasets***************************************************

#convert radiances to Bi-Directional Reflectance Factor, BRF, referred to as 'R'
#section 3.3.1.2
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

    #pretend radiance is reflectance cause that is what I'll pass in for now
    #radiance[valid_rad_idx] = radiance[valid_rad_idx] / (np.cos(np.deg2rad(SZA))[valid_rad_idx])
    #R = radiance

    return R

#calculate sun-glint flag*******************************************************
#section 3.3.2.3
def get_sun_glint_mask(solarZenith, sensorZenith, solarAzimuth, sensorAzimuth,\
                       sun_glint_exclusion_angle, sfc_ID, num_land_sfc_types):
    """
    Calculate sun-glint flag.

    [Section 3.3.2.3]
    Sun-glint water pixels are set to 0;
    non-sun-glint water pixels and land pixels are set to 1.

    Arguments:
        solarZenith {2D narray} -- Solar zenith angle in degree
        sensorZenith {2D narray} -- MAIA zenith angle in degree
        solarAzimuth {2D narray} -- Solar azimuth angle in degree
        sensorAzimuth {2D narray} -- MAIA azimuth angle in degree
        sun_glint_exclusion_angle {float} -- maximum scattering angle (degree) for sun-glint
        land_water_mask {2D binary narray} -- specify the pixel is water (0) or land (1)

    Returns:
        2D binary narray -- sunglint mask over granule same shape as solarZenith
    """

    solarZenith               = np.deg2rad(solarZenith)
    sensorZenith              = np.deg2rad(sensorZenith)
    solarAzimuth              = np.deg2rad(solarAzimuth)
    sensorAzimuth             = np.deg2rad(sensorAzimuth)
    sun_glint_exclusion_angle = np.deg2rad(sun_glint_exclusion_angle)

    cos_theta_r = np.sin(sensorZenith) * np.sin(solarZenith) \
                * np.cos(sensorAzimuth - solarAzimuth - np.pi ) + np.cos(sensorZenith) \
                * np.cos(solarZenith)
    theta_r = np.arccos(cos_theta_r)

    sun_glint_idx = np.where((theta_r >= 0) & \
                             (theta_r <= sun_glint_exclusion_angle))
    no_sun_glint_idx = np.where(~((theta_r >= 0) & \
                                  (theta_r <= sun_glint_exclusion_angle)))
    theta_r[sun_glint_idx]    = 0
    theta_r[no_sun_glint_idx] = 1
    #turn off glint calculated over land
    water = num_land_sfc_types
    theta_r[sfc_ID != water] = 1

    sun_glint_mask = theta_r
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # cmap = cm.get_cmap('PiYG', 15)
    # cmap='binary'
    # plt.figure(5)
    # plt.imshow(sun_glint_mask,cmap=cmap)
    # plt.colorbar()
    # plt.show()
    return sun_glint_mask

#calculate observables**********************************************************
#section 3.3.2.1.2
#and band center wavelengths table 2 section 2.1
#R_NIR -> bands 9, 12, 13 -> 0.86, 1.61, 1.88 micrometers
#RGB channels -> bands 6, 5, 4, respectively

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


    #make copy to not modify original memory
    R_band_6_ = np.copy(R_band_6)
    R_band_6_[R_band_6_ == -998] = -999
    bad_value = -999
    min_valid_pixels = 9
    numrows, numcols = R_band_6.shape[0], R_band_6.shape[1]
    spatial_variability_index = \
                            svi_calculation(R_band_6_, bad_value,\
                                            min_valid_pixels,\
                                            numcols, numrows)

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

#calculate bins of each pixel to query the threshold database*******************

def get_observable_level_parameter(SZA, VZA, SAA, VAA, Target_Area,\
          snow_ice_mask, sfc_ID, DOY, sun_glint_mask,\
          num_land_sfc_types):

    """
    Objective:
        calculate bins of each pixel to query the threshold database
    [Section 3.3.2.2]
    Arguments:
        SZA {2D narray} -- solar zenith angle in degrees
        VZA {2D narray} -- viewing (MAIA) zenith angle in degrees
        SAA {2D narray} -- solar azimuth angle in degrees
        VAA {2D narray} -- viewing (MAIA) azimuth angle in degrees
        Target_Area {integer} -- number assigned to target area
        land_water_mask {2D narray} -- land (1) water(0)
        snow_ice_mask {2D narray} -- no snow/ice (1) snow/ice (0)
        sfc_ID {3D narray} -- surface ID anicillary dataset for target area
        DOY {integer} -- day of year in julian calendar
        sun_glint_mask {2D narray} -- no glint (1) sunglint (0)
    Returns:
        3D narray -- 3rd axis contains 9 integers that act as indicies to query
                     the threshold database for every observable level parameter.
                     The 1st and 2cnd axes are the size of the MAIA granule.
    """
    #This is used to determine if the test should be applied over a particular
    #surface type in the get_test_determination function
    shape = np.shape(SZA)
    #define relative azimuth angle, RAZ, and cos(SZA)
    RAZ = VAA - SAA
    RAZ[RAZ<0] = RAZ[RAZ<0]*-1
    RAZ[RAZ > 180.] = -1 * RAZ[RAZ > 180.] + 360. #symmtery about principle plane
    cos_SZA = np.cos(np.deg2rad(SZA))

    #bin each input, then dstack them. return this result
    #define bins for each input
    bin_cos_SZA = np.arange(0.1, 1.1 , 0.1)
    bin_VZA     = np.arange(5. , 75. , 5.) #start at 5.0 to 0-index bin left of 5.0
    bin_RAZ     = np.arange(15., 195., 15.)
    bin_DOY     = np.arange(8. , 376., 8.0)

    binned_cos_SZA = np.digitize(cos_SZA, bin_cos_SZA, right=True)
    binned_VZA     = np.digitize(VZA    , bin_VZA, right=True)
    binned_RAZ     = np.digitize(RAZ    , bin_RAZ, right=True)
    binned_DOY     = np.digitize(DOY    , bin_DOY, right=True)

    #these datafields' raw values serve as the bins, so no modification needed:
    #Target_Area, land_water_mask, snow_ice_mask, sun_glint_mask, sfc_ID

    #put into array form to serve the whole space
    binned_DOY  = np.ones(shape) * binned_DOY
    Target_Area = np.ones(shape) * Target_Area

    #combine glint and snow-ice mask into sfc_ID
    water = num_land_sfc_types
    sfc_ID[(sun_glint_mask == 0) & (sfc_ID == water)] = num_land_sfc_types + 1
    sfc_ID[snow_ice_mask   == 0]                      = num_land_sfc_types + 2

    observable_level_parameter = np.dstack((binned_cos_SZA ,\
                                            binned_VZA     ,\
                                            binned_RAZ     ,\
                                            Target_Area    ,\
                                            sfc_ID         ,\
                                            binned_DOY     ))

    #find where there is missing data, use SZA as proxy, and give fill val
    missing_idx = np.where(SZA==-999)
    observable_level_parameter[missing_idx[0], missing_idx[1], :] = -999

    observable_level_parameter = observable_level_parameter.astype(dtype=np.int)

    return observable_level_parameter

#apply fill values to observables***********************************************
#section 3.3.2.4 in ATBD
#Using the observable level parameter (threshold bins)
#(cos_SZA, VZA, RAZ, Target Area, land_water, snowice, sfc_ID, DOY, sun_glint)
#Works on one observable at a time => function can be parallelized
#the function should return the threshold to be used for each of 5 tests at
#each pixel

#fill values
# -125 -> not applied due to surface type
# -126 -> low quality radiance
# -127 -> no data

def get_test_determination(observable_level_parameter, observable_data,\
       threshold_path, observable_name, fill_val_1, fill_val_2, fill_val_3,\
       num_land_sfc_types):

    """
    applies fill values to & finds the threshold needed at each pixel for
    for a given observable_data.
    [Section 3.3.2.4]
    Arguments:
       observable_level_parameter {3D narray} -- return from func get_observable_level_parameter()
       observable_data {2D narray} -- takes one observable at a time
       threshold_path {string} -- file path to thresholds
       observable_name {string} -- VIS_Ref, NIR_Ref, WI, NDVI, NDSI, SVI, Cirrus
       fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
       fill_val_2 {integer} -- defined in congifg file; low quality radiance
       fill_val_3 {integer} -- defined in congifg file; no data
       num_land_sfc_types {integer} -- number of kmeans land types + coast type
    Returns:
       2D narray -- observable_data with fill values applied
       2D narray -- the threshold needed at each pixel for that observable
    """
    observable_data[observable_data == -998] = fill_val_2
    observable_data[observable_data == -999] = fill_val_3

    scene_type_identifier = observable_level_parameter[:,:,4]

    # sceneID_test_configuration = np.load('./sceneID_configuration.npz')['x']#(7,21) - (tests, sceneIDs) 0 dont apply; 1 apply
    # which_test = {'VIS_Ref':0, 'NIR_Ref':1, 'WI':2, 'NDVI':3,\
    #               'NDSI':4, 'SVI':5, 'Cirrus':6}
    # which_test = which_test[observable_name]
    #
    # for current_scene_type, on_off in enumerate(sceneID_test_configuration[which_test,:]):#contains on/off sceneID config for this test
    #     if on_off == 0: #so if the test is not appplied (0) turn it off with fill_val_1
    #         observable_data[(scene_type_identifier == current_scene_type)   & \
    #                         ((observable_data != fill_val_2)                & \
    #                         (observable_data  != fill_val_3)) ]  = fill_val_1


    water     = num_land_sfc_types
    sun_glint = num_land_sfc_types + 1
    snow      = num_land_sfc_types + 2

    #apply fill values according to input observable and surface type
    if observable_name == 'VIS_Ref':
        #where water or snow/ice occur this test is not applied
        observable_data[(scene_type_identifier >= water)     & \
                        ((observable_data != fill_val_2)     & \
                         (observable_data != fill_val_3)) ]  = fill_val_1

    elif observable_name == 'NIR_Ref':
        #where land/sunglint/snow_ice occur this test is not applied
        observable_data[(scene_type_identifier != water)     & \
                        ( (observable_data != fill_val_2)    & \
                          (observable_data != fill_val_3) )] = fill_val_1

    elif observable_name == 'WI':
        #where sunglint/snow_ice occur this test is not applied
        observable_data[(scene_type_identifier >= sun_glint)   & \
                        ((observable_data != fill_val_2     )  & \
                         (observable_data != fill_val_3     )) ] = fill_val_1

    elif observable_name == 'NDVI': #this test hurts my friccin heaaaaaaaaaaaaaaaaaaaaaaaad
        #where snow_ice occurs this test is not applied
        observable_data[(scene_type_identifier == snow) &  \
                       ((observable_data != fill_val_2)  &  \
                        (observable_data != fill_val_3)) ]  = fill_val_1

        # observable_data[(scene_type_identifier >6)          &\
        #                 (scene_type_identifier !=11)          &\
        #                 (scene_type_identifier !=sun_glint) &\
        #                 (scene_type_identifier !=water)     &\
        #                ((observable_data != fill_val_2)     &\
        #                 (observable_data != fill_val_3)) ]  = fill_val_1

    elif observable_name == 'NDSI':
        # where snow_ice do not occur this test is not applied
        observable_data[(scene_type_identifier != snow)   &  \
                        ((observable_data != fill_val_2)  &  \
                         (observable_data != fill_val_3)) ]  = fill_val_1

    else:
        pass

    #Now we need to get the threshold for each pixel for one observable;
    #therefore, final return should be shape (X,Y) w/thresholds stored inside
    #and the observable_data with fill values added
    #these 2 will feed into the DTT methods

    #observable_level_parameter contains bins to query threshold database

    shape = observable_data.shape
    #make a copy; runs 10 times faster
    OLP = np.copy(observable_level_parameter)

    #pick threshold for each pixel in x by y grid
    with h5py.File(threshold_path, 'r') as hf_thresholds:
        OLP = OLP.reshape((shape[0]*shape[1], 6))
        #DOY and TA is same for all pixels in granule
        if not(np.all(OLP[:,3] == -999)) and not(np.all(OLP[:,5] == -999)):
            #not -999 index; use to define target area and day of year for the granule
            not_fillVal_idx = np.where(OLP[:,3]!=-999)
            TA  = 0#OLP[not_fillVal_idx[0], 3][0]
            DOY = OLP[not_fillVal_idx[0], 5][0]

            #put 0 index where OLP is < 0 to not raise an index error
            #then we will go back and mask the invalid thresholds as -999
            fillVal_idx = np.where((OLP == -999))# | (OLP == -9))
            OLP[fillVal_idx] = 0

            #put 0 where -9 SID appears
            invalid_SID_idx = np.where(OLP[:,4]==-9)
            OLP[invalid_SID_idx] = 0

            path = 'TA_bin_{:02d}/DOY_bin_{:02d}/{}'.format(TA, DOY, observable_name)
            print(path)
            database = hf_thresholds[path][()]

            thresholds = np.array([database[olp[0], olp[1], olp[2], olp[4]] for olp in OLP])

            thresholds[fillVal_idx[0]] = -999
            #mask over invalid SID with -999 in threshold array
            thresholds[invalid_SID_idx[0]] = -999

            #reshape to original dimensions
            thresholds = np.array(thresholds).reshape(shape)
            #mask SID -9 values as -999
            OLP = OLP.reshape((shape[0],shape[1], 6))


            return observable_data, thresholds

        return observable_data, np.ones(shape)*-999

#calculate distance to threshold************************************************
#keep fill values unchanged
#all DTT functions take thresholds and corresponding observable with fill values
#both 2D arrays of the same shape

#get_DTT_Ref_Test() also works for SVI & Cirrus
def get_DTT_Ref_Test(T, Ref, Max_valid_DTT, Min_valid_DTT, fill_val_1,\
                                                        fill_val_2, fill_val_3):

    """
    calculate the distance to threshold metric. This function is valid for
    both near infra-red and for visible reflectance test, as well as for
    spatial variability and cirrus tests.

    [Section 3.3.2.5]

    Arguments:
        T {2D narray} -- Thresholds for observable
        Ref {2D narray} -- observable; choose from vis ref, NIR ref, SVI, or Cirrus
        Max_valid_DTT {float} -- from configuration file; upper bound of DTT
        Min_valid_DTT {float} -- from configuration file; lower bound of DTT
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
        2D narray -- Distance to threshold calculated for one observable over
                     the whole space.

    """

    #max fill value of three negative fill values to know what data is valid
    max_fill_val = np.max(np.array([fill_val_1, fill_val_2, fill_val_3]))

    DTT = np.copy(Ref)
    DTT[Ref > max_fill_val] = (100 * (Ref - T) / T)[Ref > max_fill_val]
    #put upper bound on DTT (fill vals all negative)
    DTT[DTT > Max_valid_DTT]  = Max_valid_DTT
    #put lower bound on DTT where observable is valid (fill vals all negative)
    DTT[(DTT < Min_valid_DTT) & (Ref > max_fill_val)] = Min_valid_DTT

    #where T is -999 we should give a no retreival fill value (fill_val_3 = -127)
    DTT[T==-999] = fill_val_3

    return DTT

def get_DTT_NDxI_Test(T, NDxI, Max_valid_DTT, Min_valid_DTT, fill_val_1,\
                                                        fill_val_2, fill_val_3):
    """
    calculate the distance to threshold metric. This function is valid for
    NDSI test

    [Section 3.3.2.6]

    Arguments:
        T {2D narray} -- Thresholds for observable
        Ref {2D narray} -- observable; choose from NDVI, NDSI
        Max_valid_DTT {float} -- from configuration file; upper bound of DTT
        Min_valid_DTT {float} -- from configuration file; lower bound of DTT
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
        2D narray -- Distance to threshold calculated for one observable over
                     the whole space.

    """

    max_fill_val = np.max(np.array([fill_val_1, fill_val_2, fill_val_3]))
    T[T==0] = 1e-3
    DTT = np.copy(NDxI)
    DTT[NDxI > max_fill_val] = (100 * (T - np.abs(NDxI)) / T)[NDxI > max_fill_val]
    #put upper bound on DTT (fill vals all negative)
    DTT[DTT > Max_valid_DTT]  = Max_valid_DTT
    #put lower bound on DTT where observable is valid (fill vals all negative)
    DTT[(DTT < Min_valid_DTT) & (NDxI > max_fill_val)] = Min_valid_DTT

    #where T is -999 we should give a no retreival fill value (fill_val_3 = -127)
    DTT[T==-999] = fill_val_3

    return DTT


def get_DTT_NDVI_Test_over_water(T, NDxI, Max_valid_DTT, Min_valid_DTT, fill_val_1,\
                                                        fill_val_2, fill_val_3):
    """
    calculate the distance to threshold metric. This function is valid for
    NDVI test over water.

    [Section 3.3.2.6]

    Arguments:
        T {2D narray} -- Thresholds for observable
        Ref {2D narray} -- observable; choose from NDVI, NDSI
        Max_valid_DTT {float} -- from configuration file; upper bound of DTT
        Min_valid_DTT {float} -- from configuration file; lower bound of DTT
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
        2D narray -- Distance to threshold calculated for one observable over
                     the whole space.

    """
    max_fill_val = np.max(np.array([fill_val_1, fill_val_2, fill_val_3]))
    T[T==0] = 1e-3
    # DTT = np.copy(NDxI)

    #4 cases over water
    #NDxI >= 0; T >= 0
    obs_pos_T_pos_idx = np.where((T>=0) & (NDxI>=0))
    DTT_obs_pos_T_pos = (NDxI - T)/T
    #NDxI < 0; T < 0
    obs_neg_T_neg_idx = np.where((T<0) & (NDxI<0))
    DTT_obs_neg_T_neg = (T - NDxI)/T
    #NDxI >= 0; T < 0
    obs_pos_T_neg_idx = np.where((T<0) & (NDxI>=0))
    DTT_obs_pos_T_neg = (NDxI - T)/np.abs(T)
    #NDxI < 0; T >= 0
    obs_neg_T_pos_idx = np.where((T>=0) & (NDxI<0))
    DTT_obs_neg_T_pos = -1*(T - NDxI)/T

    DTT = np.zeros(np.shape(NDxI))

    DTT[obs_pos_T_pos_idx] = DTT_obs_pos_T_pos[obs_pos_T_pos_idx]
    DTT[obs_neg_T_neg_idx] = DTT_obs_neg_T_neg[obs_neg_T_neg_idx]
    DTT[obs_pos_T_neg_idx] = DTT_obs_pos_T_neg[obs_pos_T_neg_idx]
    DTT[obs_neg_T_pos_idx] = DTT_obs_neg_T_pos[obs_neg_T_pos_idx]

    #put upper bound on DTT (fill vals all negative)
    DTT[DTT > Max_valid_DTT]  = Max_valid_DTT
    #put lower bound on DTT where observable is valid (fill vals all negative)
    DTT[(DTT < Min_valid_DTT) & (NDxI > max_fill_val)] = Min_valid_DTT

    #where T is -999 we should give a no retreival fill value (fill_val_3 = -127)
    DTT[T==-999] = fill_val_3

    return DTT

def get_DTT_White_Test(T, WI, Max_valid_DTT, Min_valid_DTT, fill_val_1,\
                                                        fill_val_2, fill_val_3):

    """
    calculate the distance to threshold metric. This function is valid for
    both near infra-red and for visible reflectance test, as well as for
    spatial variability and cirrus tests.

    [Section 3.3.2.6]

    Arguments:
        T {2D narray} -- Thresholds for observable
        Ref {2D narray} -- observable; choose from whiteness index
        Max_valid_DTT {float} -- from configuration file; upper bound of DTT
        Min_valid_DTT {float} -- from configuration file; lower bound of DTT
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
        2D narray -- Distance to threshold calculated for one observable over
                     the whole space.

    """
    # print('pixels with no threshold available', np.where(T==-999)[0].shape[0]==np.where(WI < 0)[0].shape[0])

    max_fill_val = np.max(np.array([fill_val_1, fill_val_2, fill_val_3]))

    DTT = np.copy(WI)

    # print(Max_valid_DTT, Min_valid_DTT, fill_val_1,fill_val_2, fill_val_3)
    DTT[WI > max_fill_val] = (100 * (T - WI) / T)[WI > max_fill_val]
    #put upper bound on DTT (fill vals all negative)
    DTT[DTT > Max_valid_DTT]  = Max_valid_DTT
    #put lower bound on DTT where observable is valid (fill vals all negative)
    DTT[(DTT < Min_valid_DTT) & (WI > max_fill_val)] = Min_valid_DTT
    #where T is -999 we should give a no retreival fill value (fill_val_3 = -127)
    DTT[T==-999] = fill_val_3

    return DTT


#apply N tests to activate and activation value from config file****************
#the DTT files with fill values for each observable are ready

def get_cm_confidence(DTT, activation, N, fill_val_2, fill_val_3):
    """calculates final cloud mask based of the DTT, the activation value, and
       N tests needed to activate.

    [Section N/A]

    Arguments:
        DTT {3D narray} -- first 2 axies are granule dimesions, 3rd axies contains
                           DTT for each observable in this order:
                           WI, NDVI, NDSI, VIS Ref, NIR Ref, SVI, Cirrus
        activation {1D narray} -- values for each observable's DTT to exceed to
                                  be called cloudy
        N {integer} -- number of observables which have to activate to be called
                       cloudy.
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
        2D narray -- cloud mask; cloudy (0) clear(1) bad data (2) no data (3)

    """
    #create the mask only considering the activation & fill values, not N yet
    #this creates a stack of cloud masks, 7 observables deep, with fill values
    #untouched

    #cloudy_idx contains indicies where each observable independently returned
    #cloudy. The array it refers to is (X,Y,7) for 7 observables calculated over
    #the swath of MAIA
    #essentailly the format is of numpy.where() for 3D array to use later
    # [[    0.     0.     0. ...,  1265.  1265.  1267.]
    #  [   39.    42.    43. ...,   318.   319.   317.]
    #  [    0.     0.     0. ...,     6.     6.     6.]]

    #we must do it like this since each observable has a unique activation value

    #execute above comment blocks
    num_tests  = activation.size
    cloudy_idx = np.where(DTT[:,:,0] >= activation[0])
    #stack the 2D cloudy_idx with a 1D array of zeros denoting 0th element along
    #3rd axis
    zeroth_axis = np.zeros((np.shape(cloudy_idx)[1]))
    cloudy_idx = np.vstack((cloudy_idx,  zeroth_axis))

    #do the same as above in the loop but with the rest of the observables
    #which are stored along the 3rd axis
    for test_num in range(1, num_tests):
        new_cloudy_idx = np.where(DTT[:,:,test_num] >= activation[test_num])
        nth_axis       = np.ones((np.shape(new_cloudy_idx)[1])) * test_num
        new_cloudy_idx = np.vstack((new_cloudy_idx, nth_axis))
        cloudy_idx     = np.concatenate((cloudy_idx, new_cloudy_idx), axis=1)

    cloudy_idx = cloudy_idx.astype(int) #so we can index with this result

    #find indicies where fill values are
    failed_retrieval_idx = np.where(DTT == fill_val_2)
    no_data_idx          = np.where(DTT == fill_val_3)

    DTT_ = np.copy(DTT)
    DTT_[cloudy_idx[0], cloudy_idx[1], cloudy_idx[2]] = 0
    DTT_[failed_retrieval_idx]                        = 2
    DTT_[no_data_idx]                                 = 3
    #can't assign value to 'maybe_cloudy' yet since it would override 'cloudy'
    #we must check the N condition before proceeding on this

    #check N condition to distinguish 'maybe_cloudy' from 'cloudy'
    #count howmany tests returned DTT >= activation value for each pixel
    #To do this, simply add along the third axis each return type for the final
    #cloud mask, 0-3 inclusive
    #check if all tests failed with DTT = -126 <- bad quality data
    #check if all tests failed with DTT = -127 <- no data
    shape = DTT[:,:,0].shape
    cloudy_test_count      = np.zeros(shape)
    failed_retrieval_count = np.zeros(shape)
    no_data_count          = np.zeros(shape)

    for i in range(num_tests):
        cloudy_idx = np.where(DTT_[:,:,i] == 0)
        cloudy_test_count[cloudy_idx] += 1
        failed_retrieval_idx = np.where(DTT_[:,:,i] == 2)
        failed_retrieval_count[failed_retrieval_idx] += 1

        no_data_idx = np.where(DTT_[:,:,i] == 3)
        no_data_count[no_data_idx] += 1

    #populate final cloud mask; default of one assumes 'maybe cloudy' at all pixels
    final_cm    = np.ones(shape)

    cloudy_idx           = np.where(cloudy_test_count >= N)
    final_cm[cloudy_idx] = 0

    failed_retrieval_idx = np.where(failed_retrieval_count == num_tests)
    final_cm[failed_retrieval_idx] = 2

    no_data_idx = np.where(no_data_count == num_tests)
    final_cm[no_data_idx] = 3

    return final_cm

def MCM_wrapper(test_data_JPL_path, Target_Area_X, threshold_filepath,\
                sfc_ID_filepath, config_filepath, num_land_sfc_types):
    """
    simply executes function to get the final cloud mask

    [Section N/A]

    Arguments:
        test_data_JPL_path {string} -- JPL supplied data filepath; example sent to JPL
        Target_Area_X {integer} -- integer corresponding to target area
        threshold_filepath {string} -- ancillary threshold dataset filepath
        sfc_ID_filepath {string} -- ancillary surface ID dataset filepath
        config_filepath {string} -- ancillary configuration file filepath
        num_land_sfc_types {int} -- number of Kmeans land surface types; add 1 for coast


    Returns:
        Sun_glint_exclusion_angle {float} -- from configuration file in degrees;
                                             scattering angles between 0 and this
                                             are deemed as sunglint for water scenes.
        Max_RDQI {integer} -- from configuration file; denotes minimum usable
                              radiance quality flag
        Max_valid_DTT {float} -- from configuration file; upper bound of DTT
        Min_valid_DTT {float} -- from configuration file; lower bound of DTT
        fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
        fill_val_2 {integer} -- defined in congifg file; low quality radiance
        fill_val_3 {integer} -- defined in congifg file; no data
        Min_num_of_activated_tests {integer} -- same as N. Denotes min number of
                                                tests needed to activate to call
                                                pixel cloudy.
        activation {1D narray} -- values for each observable's DTT to exceed to
                                  be called cloudy
        observable_data {3D narray} -- observables stacked along 3rd axis
        DTT {3D narray} -- first 2 axies are granule dimesions, 3rd axies contains
                           DTT for each observable in this order:
                           WI, NDVI, NDSI, VIS Ref, NIR Ref, SVI, Cirrus
        final_cloud_mask {2D narray} -- cloud mask; cloudy (0) clear(1) bad data
                                                    (2) no data (3)
        BRFs {3D narray} -- reflectances stacked along third axis in this order:
                            bands 4,5,6,9,12,13
        SZA {2D narray} -- solar zenith angle in degrees
        VZA {2D narray} -- viewing (MAIA) zenith angle in degrees
        SAA {2D narray} -- solar azimuth angle in degrees
        VAA {2D narray} -- viewing (MAIA) azimuth angle in degrees
        scene_type_identifier {2D narray} -- scene ID. Values 0-28 inclusive are
                                             land types; values 29, 30, 31 are
                                             water, water with sun glint, snow/ice
                                             respectively.

    """


    start_time = time.time()
    #print('started: ' , 0)

    #get JPL provided data******************************************************
    rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12, rad_band_13,\
    RDQI_band_4, RDQI_band_5, RDQI_band_6, RDQI_band_9, RDQI_band_12, RDQI_band_13,\
    SZA, VZA, SAA, VAA,\
    d,\
    E_std_0b,\
    snow_ice_mask,\
    DOY,\
    Target_Area = get_JPL_data(test_data_JPL_path)
    # print(Target_Area, '*******************JPLMCM************')

    #define global shape for granule to process
    shape = rad_band_4.shape

    #get UIUC provided data*****************************************************
    sfc_ID,\
    Sun_glint_exclusion_angle,\
    Max_RDQI,\
    Max_valid_DTT,\
    Min_valid_DTT,\
    fill_val_1,\
    fill_val_2,\
    fill_val_3,\
    Min_num_of_activated_tests,\
    activation_values = get_UIUC_data(sfc_ID_filepath, config_filepath)

    im_scene_ID = plt.imshow(sfc_ID, vmin=0, vmax=num_land_sfc_types+1 , cmap='cubehelix')
    im_scene_ID.cmap.set_under('red')
    im_scene_ID.cmap.over('aqua')
    plt.xticks([])
    plt.yticks([])
    import sys
    sys.exit()

    #now put data through algorithm flow****************************************

    #mark bad radiance**********************************************************
    rad_band_4  = mark_bad_radiance(rad_band_4[:],  RDQI_band_4[:],  Max_RDQI)
    rad_band_5  = mark_bad_radiance(rad_band_5[:],  RDQI_band_5[:],  Max_RDQI)
    rad_band_6  = mark_bad_radiance(rad_band_6[:],  RDQI_band_6[:],  Max_RDQI)
    rad_band_9  = mark_bad_radiance(rad_band_9[:],  RDQI_band_9[:],  Max_RDQI)
    rad_band_12 = mark_bad_radiance(rad_band_12[:], RDQI_band_12[:], Max_RDQI)
    rad_band_13 = mark_bad_radiance(rad_band_13[:], RDQI_band_13[:], Max_RDQI)

    #get R**********************************************************************
    #in order MAIA  bands 6,9,4,5,12,13
    #in order MODIS bands 1,2,3,4,6 ,26
    R_band_4  = get_R(rad_band_4[:],  SZA[:], d, E_std_0b[2])
    R_band_5  = get_R(rad_band_5[:],  SZA[:], d, E_std_0b[3])
    R_band_6  = get_R(rad_band_6[:],  SZA[:], d, E_std_0b[0])
    R_band_9  = get_R(rad_band_9[:],  SZA[:], d, E_std_0b[1])
    R_band_12 = get_R(rad_band_12[:], SZA[:], d, E_std_0b[4])
    R_band_13 = get_R(rad_band_13[:], SZA[:], d, E_std_0b[5])

    BRFs = np.dstack((R_band_4,\
                      R_band_5,\
                      R_band_6,\
                      R_band_9,\
                      R_band_12,\
                      R_band_13))

    #calculate sunglint mask****************************************************
    num_land_sfc_types_plus_coast = num_land_sfc_types+1
    sun_glint_mask = get_sun_glint_mask(SZA[:], VZA[:], SAA[:], VAA[:],\
                                Sun_glint_exclusion_angle, sfc_ID, num_land_sfc_types_plus_coast)

    #calculate observables******************************************************
    #0.86, 1.61, 1.88 micrometers -> bands 9, 12, 13
    #RGB channels -> bands 6, 5, 4
    WI      = get_whiteness_index(R_band_6, R_band_5, R_band_4)
    NDVI    = get_NDVI(R_band_6, R_band_9)
    NDSI    = get_NDSI(R_band_5, R_band_12)
    VIS_Ref = get_visible_reflectance(R_band_6)
    NIR_Ref = get_NIR_reflectance(R_band_9)
    Cirrus  = get_cirrus_Ref(R_band_13)
    SVI     = get_spatial_variability_index(R_band_6, shape[0], shape[1])

    #SVI_Sfc_ID = get_spatial_variability_index(sfc_ID, shape[0], shape[1])
    #SVI = SVI - SVI_Sfc_ID
    #SVI[SVI<0] = 0
    #Cirrus[Cirrus>2] = -998
    #get observable level parameter*********************************************

    observable_level_parameter = get_observable_level_parameter(SZA[:],\
                VZA[:], SAA[:], VAA[:], Target_Area,\
                snow_ice_mask[:], sfc_ID[:], DOY, sun_glint_mask[:], num_land_sfc_types_plus_coast)

    #get test determination*****************************************************
    #combine observables into one array along third dimesnion
    observables = np.dstack((WI, NDVI, NDSI, VIS_Ref, NIR_Ref, SVI, Cirrus))
    observable_names = ['WI', 'NDVI', 'NDSI', 'VIS_Ref', 'NIR_Ref', 'SVI',\
                        'Cirrus']

    observable_data = np.empty(np.shape(observables))
    T = np.empty(np.shape(observables))
    for i in range(len(observable_names)):
        observable_data[:,:,i], T[:,:,i] = \
        get_test_determination(observable_level_parameter,\
        observables[:,:,i],\
        threshold_filepath,\
        observable_names[i],\
        fill_val_1, fill_val_2, fill_val_3, num_land_sfc_types_plus_coast)

    #retrive SID for return and DTT experiments
    scene_type_identifier = observable_level_parameter[:,:,4]

    #get DTT********************************************************************
    DTT_WI      = get_DTT_White_Test(T[:,:,0], observable_data[:,:,0], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)

    #special conditions for NDVI DTT formula
    DTT_NDVI = get_DTT_NDxI_Test(T[:,:,1] , observable_data[:,:,1], \
           Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)
    DTT_NDVI_over_water = get_DTT_NDVI_Test_over_water(T[:,:,1] , observable_data[:,:,1], \
           Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)
    #where NDVI is over water use DTT_NDVI_over_water, leave the rest
    water_SID = num_land_sfc_types_plus_coast
    water_idx = np.where(scene_type_identifier == water_SID)
    DTT_NDVI[water_idx] = DTT_NDVI_over_water[water_idx]

    #special conditions for NDSI DTT formula
    DTT_NDSI    = get_DTT_White_Test(T[:,:,2] , observable_data[:,:,2], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)

    DTT_VIS_Ref = get_DTT_Ref_Test(T[:,:,3]  , observable_data[:,:,3], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)

    DTT_NIR_Ref = get_DTT_Ref_Test(T[:,:,4]  , observable_data[:,:,4], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)

    DTT_SVI     = get_DTT_Ref_Test(T[:,:,5]  , observable_data[:,:,5], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)

    DTT_Cirrus  = get_DTT_Ref_Test(T[:,:,6]  , observable_data[:,:,6], \
               Max_valid_DTT, Min_valid_DTT, fill_val_1, fill_val_2, fill_val_3)


    DTT = np.dstack((DTT_WI     ,\
                     DTT_NDVI   ,\
                     DTT_NDSI   ,\
                     DTT_VIS_Ref,\
                     DTT_NIR_Ref,\
                     DTT_SVI    ,\
                     DTT_Cirrus))


    #in order the activation values are
    #WI, NDVI, NDSI, VIS Ref, NIR Ref, SVI, Cirrus
    #I reformat the values as such since the code handles each observable
    #independently, even if two observables belong to the same test
    activation_values = np.array([activation_values[0],\
                                  activation_values[1],\
                                  activation_values[1],\
                                  activation_values[2],\
                                  activation_values[2],\
                                  activation_values[3],\
                                  activation_values[4]])

    final_cloud_mask = get_cm_confidence(DTT, activation_values,\
                             Min_num_of_activated_tests, fill_val_2, fill_val_3)

    print('finished: ' , time.time() - start_time)




    return Sun_glint_exclusion_angle,\
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
           T


#badda bing badda boom... cloud mask********************************************

if __name__ == '__main__':
    pass
