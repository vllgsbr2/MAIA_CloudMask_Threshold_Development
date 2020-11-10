from calc_observables import *
from calc_OLP import *
import h5py
from fetch_MCM_input_data import *
import numpy as np


def get_data(mcm_input_file):
    '''for thresh i need to get OLP/obs/CM then do historgram approach
    input file has everything that I need'''

    with h5py.File(mcm_input_file, 'r') as JPL_file:
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

        snow_ice_mask = JPL_file['Anicillary_Geometric_Product/Snow_Ice_Mask/Snow_Ice_Mask']

        DOY = JPL_file.get('Anicillary_Radiometric_Product/Day_of_year/Day_of_year')[()]

        Target_Area = JPL_file.get('Anicillary_Radiometric_Product/Target_Area/Target_Area')[()]

        mod35cm = JPL_file.get('MOD35_cloud_mask')[()]

    return rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12, rad_band_13,\
           RDQI_b4, RDQI_b5, RDQI_b6, RDQI_b9, RDQI_b12, RDQI_b13,\
           SZA, VZA, SAA, VAA,\
           d,\
           E_std_0,\
           snow_ice_mask,\
           DOY,\
           Target_Area,\
           mod35cm

def get_obs(rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12,\
            rad_band_13, SZA, VZA, SAA, VAA, d, E_std_0, snow_ice_mask, DOY,\
            Target_Area):

    R_band_4  = get_R(rad_band_4 , SZA, d, E_std_0b[2])
    R_band_5  = get_R(rad_band_5 , SZA, d, E_std_0b[3])
    R_band_6  = get_R(rad_band_6 , SZA, d, E_std_0b[0])
    R_band_9  = get_R(rad_band_9 , SZA, d, E_std_0b[1])
    R_band_12 = get_R(rad_band_12, SZA, d, E_std_0b[4])
    R_band_13 = get_R(rad_band_13, SZA, d, E_std_0b[5])

    whiteness_index           = get_whiteness_index(R_band_6, R_band_5, R_band_4)
    NDVI                      = get_NDVI(R_band_6, R_band_9)
    NDSI                      = get_NDSI(R_band_5, R_band_12)
    visible_reflectance       = get_visible_reflectance(R_band_6)
    NIR_reflectance           = get_NIR_reflectance(R_band_9)
    spatial_variability_index = get_spatial_variability_index(R_band_6)
    cirrus_Ref                = get_cirrus_Ref(R_band_13)

    return whiteness_index,\
           NDVI,\
           NDSI,\
           visible_reflectance,\
           NIR_reflectance,\
           spatial_variability_index,\
           cirrus_Ref

def get_SID(sfc_ID_home, DOY):
    DOY       = int(DOY)
    DOY_bins  = np.arange(8,376,8)
    DOY_bin   = np.digitize(DOY, DOY_bins, right=True)
    DOY_end   = (DOY_bin+1)*8

    sfc_ID_filepath    = '{}/surfaceID_{}_{:03d}.nc'.format(sfc_ID_home, PTA, DOY_end)
    with Dataset(sfc_ID_path, 'r') as nc_SID:
        SID = nc_SID.variables['surface_ID'][:,:]
    #overlay SIM and SGM on SID to make scene ID and rename to SID
    SID[(sun_glint_mask  == 0) & (SID == 12)]  = 13
    SID[ snow_ice_mask   == 0]                 = 14

    return SID

def get_observable_level_parameter(SZA, VZA, SAA, VAA, Target_Area, SID,\
                                   snow_ice_mask, DOY, sun_glint_mask):

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
    bin_VZA     = np.arange(5 , 75 , 5) #start at 5.0 to 0-index bin left of 5.0
    bin_RAZ     = np.arange(15, 195, 15)
    bin_DOY     = np.arange(8 , 376, 8)

    binned_cos_SZA = np.digitize(cos_SZA, bin_cos_SZA, right=True)
    binned_VZA     = np.digitize(VZA    , bin_VZA, right=True)
    binned_RAZ     = np.digitize(RAZ    , bin_RAZ, right=True)

    binned_DOY     = np.digitize(DOY    , bin_DOY, right=True)
    DOY_end = (binned_DOY+1)*8

    #these datafields' raw values serve as the bins, so no modification needed:
    #Target_Area, snow_ice_mask, sun_glint_mask, sfc_ID

    #put into array form to serve the whole space
    binned_DOY  = np.ones(shape) * binned_DOY
    Target_Area = np.ones(shape) * Target_Area

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


def group_data(OLP, mod35cm, WI, NDVI, NDSI, visRef, NIRRef, SVI, cirrus):
    """
    Objective:
        Group data by observable_level_paramter (OLP), such that all data in same
        group has a matching OLP. The data point is stored in the group with its
        observables, cloud mask, time stamp, and row/col. Will process one MAIA
        grid at a time. No return, will just be written to file in this function.
    Return:
        void
    """
    # observable_level_parameter = np.dstack((cos_SZA
    #                                         VZA
    #                                         RAZ
    #                                         TA
    #                                         sfc_ID
    #                                         DOY     ))

    obs = np.dstack((WI, NDVI, NDSI, visRef, NIRRef, SVI, cirrus))
    shape = CM.shape
    row_col_product = shape[0]*shape[1]
    OLP = OLP.reshape(row_col_product, 6)
    obs = obs.reshape(row_col_product, 7)
    CM  = CM.reshape(row_col_product)

    #remove empty data points
    #where cos(SZA) is negative (which is not possible)
    full_idx = np.where(OLP[:,0] !=-999) # obs -> (1, 1e6-x)

    OLP = OLP[full_idx[0], :]
    obs = obs[full_idx[0], :]
    CM  = CM[full_idx[0]]

    #now again for the -998 in the cloud mask
    full_idx = np.where(CM !=-998) # obs -> (1, 1e6-x)

    OLP = OLP[full_idx[0], :]
    obs = obs[full_idx[0], :]
    CM  = CM[full_idx[0]]

    thresh_dict = {}

    #now for any OLP combo, make a group and save the data points into it
    for i in range(CM.shape[0]):
        #0 cosSZA
        #1 VZA
        #2 RAZ
        #3 TA
        #4 Scene_ID
        #5 DOY
        temp_OLP = OLP[i,:]
        group = 'cosSZA_{:02d}_VZA_{:02d}_RAZ_{:02d}_TA_{:02d}_sceneID_{:02d}_DOY_{:02d}'\
                .format(temp_OLP[0], temp_OLP[1], temp_OLP[2],\
                        1, temp_OLP[4], temp_OLP[5])

        data = np.array([CM[i]   ,\
                         obs[i,0],\
                         obs[i,1],\
                         obs[i,2],\
                         obs[i,3],\
                         obs[i,4],\
                         obs[i,5],\
                         obs[i,6] ])

        #add key with empty list, then populate it.
        thresh_dict.setdefault(group, [])
        thresh_dict[group].append(data)

    with h5py.File('./grouped_obs_&_CM_one_scene', 'w') as hf_group:

        for key, val in thresh_dict.items():
            try:
                hf_group.create_dataset(key, data=np.array(val), maxshape=(None,8))
            except:
                group_shape = hf_group[key].shape[0]
                hf_group[key].resize(group_shape + np.array(val).shape[0], axis=0)
                hf_group[key][group_shape:, :] = np.array(val)

def calc_thresh(group_file):
def calc_thresh(thresh_home, group_file, DOY_bin, TA):
    '''
    Objective:
        Takes in grouped_obs_and_CM.hdf5 file. Inside are a datasets for
        a bin and inside are rows containing the cloud mask and
        observables for each pixel. The OLP is in the dataset name. The
        threshold is then calculated for that dataset and saved into a
        threshold file.
    Arguments:
        group_file {str} -- contains data points to calc threshold for
        all bins in the file
    Return:
        void
    '''

    DOY_end   = (DOY_bin+1)*8
    DOY_start = DOY_end - 7

    fill_val = -998

    num_samples_valid_hist = 0

    with h5py.File(group_file, 'r') as hf_group,\
         h5py.File(thresh_home + '/thresholds_DOY_{:03d}_to_{:03d}_bin_{:02d}.h5'.format(DOY_start, DOY_end, DOY_bin), 'w') as hf_thresh:

        #cosSZA_00_VZA_00_RAZ_00_TA_00_sceneID_00_DOY_00
        TA_group  = hf_thresh.create_group('TA_bin_{:02d}'.format(TA))
        DOY_group = TA_group.create_group('DOY_bin_{:02d}'.format(DOY_bin))

        num_sfc_types = 15

        master_thresholds = np.ones((10*15*12*num_sfc_types)).reshape((10,15,12,num_sfc_types))*-999
        obs_names = ['WI', 'NDVI', 'NDSI', 'VIS_Ref', 'NIR_Ref', 'SVI', 'Cirrus']
        for obs in obs_names:
            DOY_group.create_dataset(obs, data=master_thresholds)

        hf_keys    = list(hf_group.keys())
        num_points = len(hf_keys)

        for count, bin_ID in enumerate(hf_keys):
            #location in array to store threshold (cos(SZA), VZA, RAZ, Scene_ID)
            bin_idx = [int(bin_ID[7:9]), int(bin_ID[14:16]), int(bin_ID[21:23]), int(bin_ID[38:40])]

            # #only calc a thresh when valid surface ID is available
            # #invalid is -9
            # if bin_idx[3] != -9:

            cloud_mask = hf_group[bin_ID][:,0].astype(dtype=np.int)
            obs        = hf_group[bin_ID][:,1:]


            sfc_ID_bin = bin_idx[3]#12 is water no glint

            clear_idx  = np.where((cloud_mask != 0) & (cloud_mask > fill_val))
            clear_obs  = obs[clear_idx[0],:]

            cloudy_idx = np.where((cloud_mask == 0) & (cloud_mask > fill_val))
            cloudy_obs = obs[cloudy_idx[0],:]


            for i in range(7):
                #path to TA/DOY/obs threshold dataset
                path = 'TA_bin_{:02d}/DOY_bin_{:02d}/{}'.format(TA, DOY_bin , obs_names[i])
                # print(path)

                #clean the obs for the thresh calculation
                clean_clear_obs = clear_obs[:,i]
                clean_clear_obs = clean_clear_obs[(clean_clear_obs > -998) & (clean_clear_obs <= 32767)]

                clean_cloudy_obs = cloudy_obs[:,i]
                clean_cloudy_obs = clean_cloudy_obs[(clean_cloudy_obs > -998) & (clean_cloudy_obs <= 32767)]

                #WI
                if i==0:
                    if clean_clear_obs.shape[0] > num_samples_valid_hist:
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] = \
                        np.nanpercentile(clean_clear_obs, 1)


                #NDxI
                #pick max from cloudy hist
                elif i==1 or i==2:
                    if clean_cloudy_obs.shape[0] > num_samples_valid_hist:
                        hist, bin_edges = np.histogram(clean_cloudy_obs, bins=128, range=(-1,1))
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] =\
                        bin_edges[1:][hist==hist.max()].min()

                #VIS/NIR/SVI/Cirrus
                else:
                    if clean_clear_obs.shape[0] > num_samples_valid_hist:
                        current_thresh = np.nanpercentile(clean_clear_obs, 99)
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2],\
                                        bin_idx[3]] = current_thresh

                current_thresh = hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]]
                if np.abs(current_thresh) > 2:
                    debug_line = 'cos(SZA): {:02d} VZA: {:02d} RAA: {:02d} SID: {:02d} thresh: {:3.3f}'.\
                              format(bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3], current_thresh)
                    print(debug_line)



if __name__ == '__main__':
    import configparser

    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA      = config['current PTA']['PTA']
    PTA_path = config['PTAs'][PTA]
    MCM_input_filepath = config['supporting directories']['MCM_Input']
    MCM_input_filepaths = [MCM_input_filepath + '/' + x for x in os.listdir(MCM_input_filepath)]
    mcm_input_file = MCM_input_filepaths[0]

    #grab data out of input file
    rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12, rad_band_13,\
    RDQI_b4, RDQI_b5, RDQI_b6, RDQI_b9, RDQI_b12, RDQI_b13,\
    SZA, VZA, SAA, VAA,\
    d,\
    E_std_0,\
    snow_ice_mask,\
    DOY,\
    Target_Area,\
    mod35cm = get_JPL_data(mcm_input_file)

    #get obs
    WI,\
    NDVI,\
    NDSI,\
    visRef,\
    NIRRef,\
    SVI,\
    cirrus = get_obs(rad_band_4, rad_band_5, rad_band_6, rad_band_9, rad_band_12,\
                rad_band_13, SZA, VZA, SAA, VAA, d, E_std_0, snow_ice_mask, DOY,\
                Target_Area)

    #get sunglint mask
    sun_glint_mask = get_sun_glint_mask(SZA, VZA, SAA, VAA,\
                                     sun_glint_exclusion_angle, land_water_mask)
    #get sfcID with sunglin mask and snow ice mask incorporated
    #land/water is decided by the MAIA AGP file built into the SID k means cluster code
    SID = get_SID(sfc_ID_home, DOY)

    #get OLP
    OLP = get_observable_level_parameter(SZA, VZA, SAA, VAA, Target_Area, SID,\
                                         snow_ice_mask, DOY, sun_glint_mask)

    #group data
    #writes group to file
    group_data(OLP, mod35cm, WI, NDVI, NDSI, visRef, NIRRef, SVI, cirrus)

    #get thresholds
