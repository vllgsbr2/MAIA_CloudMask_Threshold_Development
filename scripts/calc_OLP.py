import numpy as np

def add_sceneID_MOD03_SFCTYPES(observable_level_parameter, num_land_sfc_types, MOD03_sfctypes):

        """
        helper function to combine water/sunglint/snow-ice mask/sfc_ID into
        one mask. This way the threhsolds can be retrieved with less queries.
        [Section N/A]
        Arguments:
            observable_level_parameter {3D int narray} -- return from func get_observable_level_parameter()
            num_land_sfc_types {int} -- number of land surface types from max BRF clusters
            MOD03_sfctypes {2D int narray} -- 0-7 water (1 land type) surface types from MOD03 MODIS TERRA product
        Returns:
            2D narray -- scene ID. Values 0-28 inclusive are land types; values
                         29, 30, 31 are water, water with sun glint, snow/ice
                         respectively. Is the size of the granule. These integers
                         serve as the indicies to select a threshold based off
                         surface type.
        """
        # land_water_bins {2D narray} -- land (1) water(0)
        # sun_glint_bins {2D narray} -- no glint (1) sunglint (0)
        # snow_ice_bins {2D narray} -- no snow/ice (1) snow/ice (0)

        #over lay water/glint/snow_ice onto sfc_ID to create a scene_type_identifier
        land_water_bins = observable_level_parameter[:,:, 4]
        sun_glint_bins  = observable_level_parameter[:,:,-1]
        snow_ice_bins   = observable_level_parameter[:,:, 5]

        sfc_ID_bins = observable_level_parameter[:,:,6]
        scene_type_identifier = sfc_ID_bins

        #MOD03_sfctypes
        #0-shallow ocean
        #1-land
        #2-ocean/lake coast
        #3-shallow inland water
        #4-seasonal inland water
        #5-deep inland water
        #6-moderate continental ocean
        #7-deep ocean
        scene_type_identifier[MOD03_sfctypes==0] = num_land_sfc_types + 2
        scene_type_identifier[MOD03_sfctypes==2] = num_land_sfc_types + 3
        scene_type_identifier[MOD03_sfctypes==3] = num_land_sfc_types + 4
        scene_type_identifier[MOD03_sfctypes==4] = num_land_sfc_types + 5
        scene_type_identifier[MOD03_sfctypes==5] = num_land_sfc_types + 6
        scene_type_identifier[MOD03_sfctypes==6] = num_land_sfc_types + 7
        scene_type_identifier[MOD03_sfctypes==7] = num_land_sfc_types + 8

        scene_type_identifier[(sun_glint_bins  == 0) & \
                              (MOD03_sfctypes != 1) ]  = num_land_sfc_types + 0
        scene_type_identifier[ snow_ice_bins   == 0]   = num_land_sfc_types + 1

        OLP = np.zeros((1000,1000,6))
        OLP[:,:,:4] = observable_level_parameter[:,:,:4]#cosSZA, VZA, RAZ, TA
        OLP[:,:,4]  = scene_type_identifier             #scene_ID
        OLP[:,:,5] = observable_level_parameter[:,:,7]  #DOY



        return OLP

def get_observable_level_parameter_MOD03_SFCTYPES(SZA, VZA, SAA, VAA, Target_Area,\
          land_water_mask, snow_ice_mask, DOY, sun_glint_mask, time_stamp,\
          num_land_sfc_types, MOD03_sfctypes):

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
    if DOY_end > 360:
        DOY_sfcID = 360
    else:
        DOY_sfcID = DOY_end
    sfc_ID_path = home + 'LA_surface_types/surfaceID_LA_{:03d}.nc'.format(DOY_sfcID)
    sfc_ID = Dataset(sfc_ID_path, 'r').variables['surface_ID'][:,:]

    #these datafields' raw values serve as the bins, so no modification needed:
    #Target_Area, land_water_mask, snow_ice_mask, sun_glint_mask, sfc_ID

    #put into array form to serve the whole space
    binned_DOY  = np.ones(shape) * binned_DOY
    Target_Area = np.ones(shape) * Target_Area

    observable_level_parameter = np.dstack((binned_cos_SZA ,\
                                            binned_VZA     ,\
                                            binned_RAZ     ,\
                                            Target_Area    ,\
                                            land_water_mask,\
                                            snow_ice_mask  ,\
                                            sfc_ID         ,\
                                            binned_DOY     ,\
                                            sun_glint_mask))

    observable_level_parameter = add_sceneID_MOD03_SFCTYPES(observable_level_parameter,\
                                             num_land_sfc_types, MOD03_sfctypes)

    #find where there is missing data, use SZA as proxy, and give fill val
    missing_idx = np.where(SZA==-999)
    observable_level_parameter[missing_idx[0], missing_idx[1], :] = -999

    observable_level_parameter = observable_level_parameter.astype(dtype=np.int)

    return observable_level_parameter

if __name__ == '__main__':

    import h5py
    import mpi4py.MPI as MPI
    import tables
    from netCDF4 import Dataset
    import os
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:
            #open database to read
            home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
            PTA_file_path = home + 'LA_database_60_cores/'
            database_files = os.listdir(PTA_file_path)
            database_files = [PTA_file_path + filename for filename in database_files]
            database_files = np.sort(database_files)
            hf_database_path = database_files[r]

            with h5py.File(hf_database_path, 'r') as hf_database:

                len_pta       = len(PTA_file_path)
                start, end    = hf_database_path[len_pta + 26:len_pta +31], hf_database_path[len_pta+36:len_pta+41]

                #create/open hdf5 file to store observables
                PTA_file_path_OLP = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
                hf_OLP_path = '{}OLP_database_60_cores/LA_PTA_OLP_start_{}_end_{}_.hdf5'.format(PTA_file_path_OLP, start, end)

                hf_database_keys = list(hf_database.keys())
                observables = ['WI', 'NDVI', 'NDSI', 'visRef', 'nirRef', 'SVI', 'cirrus']

                with h5py.File(hf_OLP_path, 'w') as hf_OLP:

                    for time_stamp in hf_database_keys:

                        PTA_file_path = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data'
                        hf_OLP_path   = '{}/LA_PTA_OLP_start_{}_end_{}_.hdf5'.format(PTA_file_path, start, end)

                        SZA = hf_database[time_stamp+'/sunView_geometry/solarZenith'][()]
                        VZA = hf_database[time_stamp+'/sunView_geometry/sensorZenith'][()]
                        VAA = hf_database[time_stamp+'/sunView_geometry/sensorAzimuth'][()]
                        SAA = hf_database[time_stamp+'/sunView_geometry/solarAzimuth'][()]
                        TA  = 0 #will change depending where database is stored
                        LWM = hf_database[time_stamp+'/cloud_mask/Land_Water_Flag'][()]
                        SIM = hf_database[time_stamp+'/cloud_mask/Snow_Ice_Background_Flag'][()]
                        DOY = time_stamp[4:7]
                        SGM = hf_database[time_stamp+'/cloud_mask/Sun_glint_Flag'][()]
                        num_land_sfc_types = 12 #read from config file later                        
                        MOD03_sfctypes     = hf_database[time_stamp+'/MOD03_LandSeaMask'][()]

                        #OLP = get_observable_level_parameter(SZA, VZA, SAA,\
                        #      VAA, TA, LWM, SIM, sfc_ID_LAday48, DOY, SGM, time_stamp)
                        OLP = get_observable_level_parameter_MOD03_SFCTYPES(SZA, VZA, SAA, VAA, TA,\
                                            LWM, SIM, DOY, SGM, time_stamp,\
                                            num_land_sfc_types, MOD03_sfctypes)

                        try:
                            group = hf_OLP.create_group(time_stamp)
                            group.create_dataset('observable_level_paramter', data=OLP, compression='gzip')
                        except:
                            try:
                                group.create_dataset('observable_level_paramter', data=OLP, compression='gzip')
                                hf_OLP[time_stamp+'/observable_level_paramter'][:] = OLP
                            except:
                                hf_OLP[time_stamp+'/observable_level_paramter'][:] = OLP
