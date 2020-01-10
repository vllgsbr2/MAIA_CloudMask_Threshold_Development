import numpy as np
import h5py

#all data is 1000x1000 pixels with a -999 fill value. Every pixel in the grid
#corresponds to pixels in every other grid i.e. lat/lon/cm/redband at [0,0] belong
#to the same data point

def get_observable_level_parameter(SZA, VZA, SAA, VAA, Target_Area,\
          land_water_mask, snow_ice_mask, sfc_ID, DOY, sun_glint_mask):

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

    #This is used todetermine if the test should be applied over a particular
    #surface type in the get_test_determination function

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
    sfc_ID         = sfc_ID[:,:,binned_DOY] #just choose the day for sfc_ID map

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

    observable_level_parameter = observable_level_parameter.astype(dtype=np.int)

    return observable_level_parameter

def get_observables():
    

def get_test_determination(observable_level_parameter, observable_data,\
       threshold_database, observable_name, fill_val_1, fill_val_2, fill_val_3):

    """
    Objective:
        applies fill values to & finds the threshold needed at each pixel for a
        given observable_data.

    [Section 3.3.2.4]

    Arguments:
       observable_level_parameter {3D narray} -- return from func get_observable_level_parameter()
       observable_data {2D narray} -- takes one observable at a time
       threshold_database {6D narray} -- database for the specific observable
       observable_name {string} -- VIS_Ref, NIR_Ref, WI, NDVI, NDSI, SVI, Cirrus
       fill_val_1 {integer} -- defined in congifg file; not applied due to surface type
       fill_val_2 {integer} -- defined in congifg file; low quality radiance
       fill_val_3 {integer} -- defined in congifg file; no data

    Returns:
       2D narray -- observable_data with fill values applied
       2D narray -- the threshold needed at each pixel for that observable

    """
    observable_data[observable_data == -998] = fill_val_2
    observable_data[observable_data == -999] = fill_val_3

    land_water_bins = observable_level_parameter[:,:,4]
    snow_ice_bins   = observable_level_parameter[:,:,5]
    sun_glint_bins  = observable_level_parameter[:,:,8]

    #apply fill values according to input observable and surface type
    if observable_name == 'VIS_Ref':
        #where water or snow/ice occur this test is not applied
        observable_data[((land_water_bins == 0) | (snow_ice_bins == 0)) &     \
                        ((observable_data != fill_val_2) &                    \
                         (observable_data != fill_val_3)) ] = fill_val_1

    elif observable_name == 'NIR_Ref':
        #where land/sunglint/snow_ice occur this test is not applied
        observable_data[((sun_glint_bins  == 0)  | \
                         (land_water_bins == 1)  | \
                         (snow_ice_bins   == 0)) & \
                        ((observable_data != fill_val_2) & \
                         (observable_data != fill_val_3))]    = fill_val_1

    elif observable_name == 'WI':
        #where sunglint/snow_ice occur this test is not applied
        observable_data[((sun_glint_bins == 0)           |  \
                         (snow_ice_bins  == 0))          &  \
                        ((observable_data != fill_val_2) &  \
                         (observable_data != fill_val_3)) ]    = fill_val_1

    elif observable_name == 'NDVI':
        #where snow_ice occurs this test is not applied
        observable_data[(snow_ice_bins == 0)             &  \
                       ((observable_data != fill_val_2)  &  \
                        (observable_data != fill_val_3)) ]    = fill_val_1

    elif observable_name == 'NDSI':
        #where snow_ice do not occur this test is not applied
        observable_data[(snow_ice_bins == 1)             &  \
                        ((observable_data != fill_val_2) &  \
                         (observable_data != fill_val_3)) ]    = fill_val_1

    else:
        pass

    #Now we need to get the threshold for each pixel for one observable;
    #therefore, final return should be shape (X,Y) w/thresholds stored inside
    #and the observable_data with fill values added
    #these 2 will feed into the DTT methods

    #observable_level_parameter contains bins to query threshold database
    rows, cols = np.arange(shape[0]), np.arange(shape[1])

    #combine water/sunglint/snow-ice mask/sfc_ID into one mask
    #This way the threhsolds can be retrieved with less queries
    def make_sceneID(observable_level_parameter,land_water_bins,\
                     sun_glint_bins, snow_ice_bins):

        """
        helper function to combine water/sunglint/snow-ice mask/sfc_ID into
        one mask. This way the threhsolds can be retrieved with less queries.

        [Section N/A]

        Arguments:
            observable_level_parameter {3D narray} -- return from func get_observable_level_parameter()
            land_water_bins {2D narray} -- land (1) water(0)
            sun_glint_bins {2D narray} -- no glint (1) sunglint (0)
            snow_ice_bins {2D narray} -- no snow/ice (1) snow/ice (0)

        Returns:
            2D narray -- scene ID. Values 0-28 inclusive are land types; values
                         29, 30, 31 are water, water with sun glint, snow/ice
                         respectively. Is the size of the granule. These integers
                         serve as the indicies to select a threshold based off
                         surface type.

        """

        #over lay water/glint/snow_ice onto sfc_ID to create a scene_type_identifier
        sfc_ID_bins = observable_level_parameter[:,:,6]
        scene_type_identifier = sfc_ID_bins

        scene_type_identifier[land_water_bins == 0]     = 30
        scene_type_identifier[(sun_glint_bins == 1) & \
                              (land_water_bins == 0)]   = 31
        scene_type_identifier[snow_ice_bins == 0]       = 32

        return scene_type_identifier

    scene_type_identifier = make_sceneID(observable_level_parameter,land_water_bins,\
                     sun_glint_bins, snow_ice_bins)

    #because scene_type_identifier (sfc_ID) contains information on
    #sunglint/snow_ice/water/land we use less dimensions to decribe the scene
    T = [[threshold_database[observable_level_parameter[i,j,0],\
                             observable_level_parameter[i,j,1],\
                             observable_level_parameter[i,j,2],\
                             observable_level_parameter[i,j,3],\
                             scene_type_identifier[i,j]            ,\
                             observable_level_parameter[i,j,7] ]\
                             for i in rows] for j in cols]
    return observable_data, T

def make_sceneID(land_water_bins, sun_glint_bins, snow_ice_bins, sfc_ID):

    """
    helper function to combine water/sunglint/snow-ice mask/sfc_ID into
    one mask. This way the threhsolds can be retrieved with less queries.

    [Section N/A]

    Arguments:
        land_water_bins {2D narray} -- land (1) water(0)
        sun_glint_bins {2D narray} -- no glint (1) sunglint (0)
        snow_ice_bins {2D narray} -- no snow/ice (1) snow/ice (0)

    Returns:
        2D narray -- scene ID. Values 0-28 inclusive are land types; values
                     29, 30, 31 are water, water with sun glint, snow/ice
                     respectively. Is the size of the granule. These integers
                     serve as the indicies to select a threshold based off
                     surface type.

    """

    #over lay water/glint/snow_ice onto sfc_ID to create a scene_type_identifier
    scene_type_identifier = sfc_ID

    scene_type_identifier[land_water_bins == 0]     = 29
    scene_type_identifier[(sun_glint_bins == 1) & \
                          (land_water_bins == 0)]   = 30
    scene_type_identifier[snow_ice_bins == 0]       = 31

    return scene_type_identifier

#(cos_SZA, VZA, RAZ, Target Area, land_water, snowice, sfc_ID, DOY, sun_glint)
#(cos_SZA, VZA, RAZ, Target Area, scene_ID, DOY)
#(10,14,12,30,33,46)

#OLP calculates the bins from the givens
#Test_determination creates the scene_ID and queries the threshold array

#what I need is a way to get the proper index and then a way to populate the
#threshold into the array

#If I get a thresholds for example, I would take the threshold and the bins used
#to retrieve it. Then I can assign it directly into the database.

#for all VZA/cos(SZA)/RAZ/DOY for one PTA, for one scene_ID(SGM,LWM,SIM)
#later we can choose a particular DOY and sun-view geometry

def calc_threshold_test(observable, observable_level_paramter, target_area, sfc_ID, \
  file_path = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_PTA_Subsets.HDF5'):
    '''
    Objective:
        Calculate the threhsolds based on clear data points based on a particular
        observable and observable level parameters.

    Arguments:
        observable {string} -- name of observable
        observable_level_paramter {1D array} -- only use data points that fall
                                                fall wihtin these bins. In order
                                                they are:
                                                [cosSZA,VZA,RAZ,TA,Scene_ID,DOY]
        target_area {int} -- integer denoting what the target area is.
                             (no final list onto which target is what number)
        sfc_ID {2D narray} -- pass in the sfc_ID map corresponding to the target
        sfcID_lat, sfcID_lon {2D narray} -- lat/lon of sfc_ID map
        file_path {str} -- path to database from create_database.py

    Return:
        Threshold {float} -- where 1% of clear data points misclassfied, as in
                             MOD35 visible reflectance test.
        cloudy_count, clear_count {int} -- number of MOD35 confident cloudy data
                                           points, all other classes of clear
    '''

    import numpy as np
    import h5py
    import sys
    sys.path.insert(0,'/Users/vllgsbr2/Desktop/MAIA_JPL_code/JPL_MAIA_CM')
    from JPL_MCM import get_whiteness_index, get_NDVI, get_NDSI, get_visible_reflectance,\
    get_NIR_reflectance, get_cirrus_Ref, get_sun_glint_mask, get_observable_level_parameter
    from svi_dynamic_size_input import svi_calculation


    MODIS_database = h5py.File(file_path, 'r')
    time_stamps = list(MODIS_database.keys())

    cloudy_count = 0
    clear_count  = 0

    for epoch, time_stamp in enumerate(time_stamps):
        #retrieve cloud mask for each pixel
        cloud_mask = MODIS_database[time_stamp+'/cloud_mask/Unobstructed_FOV_Quality_Flag'][:]
        #confident cloudy 0; all else 1
        cloud_mask[cloud_mask != 0] = 1
        cloud_mask = cloud_mask.flatten()

        cloudy_count  += len(np.where(cloud_mask==0)[0])
        clear_count   += len(np.where(cloud_mask==1)[0])
        cloud_fraction = (len(np.where(cloud_mask==0)[0])/cloud_mask.size)#*100


        #grab observable level parameters from MODIS
        #->(cosSZA, VZA, RAZ, TA, scene_ID, DOY)

        #grab land_water/sunglint/snow-ice mask
        LWM = MODIS_database[time_stamp+'/cloud_mask/Land_Water_Flag'][:]
        SIM = MODIS_database[time_stamp+'/cloud_mask/Snow_Ice_Background_Flag'][:]
        SGM = MODIS_database[time_stamp+'/cloud_mask/Sun_glint_Flag'][:]

        #grab obeservable level paramters
        SZA    = MODIS_database[time_stamp+'/sunView_geometry/solarZenith'][:]
        cosSZA = np.cos(np.deg2rad(SZA))
        VZA    = MODIS_database[time_stamp+'/sunView_geometry/sensorZenith'][:]
        VAA    = MODIS_database[time_stamp+'/sunView_geometry/sensorAzimuth'][:]
        SAA    = MODIS_database[time_stamp+'/sunView_geometry/solarAzimuth'][:]
        RAZ    = VAA - 180.0 - SAA
        TA     = target_area

        #grab MODIS lat/lon
        MOD03_lat = MODIS_database[time_stamp+'/geolocation/lat'][:]
        MOD03_lon = MODIS_database[time_stamp+'/geolocation/lon'][:]

        #calculate observable level parameter for the scence ID
        #Note this is the retrieved OLP to compare with OLP passed into this
        #function.

        #to get sfc_ID for OLP we need to do a nearest neighbor search between
        #the sfc_ID lat/lon and the MODIS lat/lon
        def regrid_sfcID_2_MODIS(MOD03_lat, MOD03_lon, sfcID_lat, sfcID_lon):
            '''
            Ojective:
                Regrid the sfc_ID (on the MAIA grid), to the MODIS using nearest
                neighbor in lat/lon space with some distance threshold. Only the
                intersection of of points are valid.
            Arguments:
                MOD03_lat, MOD03_lon {2D narray} -- lat/lon corresponding to database
                sfcID_lat, sfcID_lon {2D narray} -- lat/lon corresponding to sfc_ID
            Return:
                regridded_latlon {2D narray} -- idk yet lmao
            '''
            pass

        Scene_ID = make_sceneID(LWM, SGM, SIM, sfc_ID)
        OLP = get_observable_level_parameter(SZA, VZA, SAA, VAA, TA, Scene_ID, DOY)

        #use output from regriding function to only selcect points valid for the
        #sfc_ID available

        #convert MODIS Reflectance into BRF
        #RGB; NIR_X - 1,2,3 = NDVI,NDSI,Cirrus
        if observable=='WI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA
            Ref_grn  = MODIS_database[time_stamp+'/reflectance/band_4'][:]/cosSZA
            Ref_blu  = MODIS_database[time_stamp+'/reflectance/band_3'][:]/cosSZA

            #flatten output to count instances of bins
            obs = get_whiteness_index(Ref_red, Ref_grn, Ref_blu).flatten()

        elif observable=='NDVI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA
            Ref_NIR1 = MODIS_database[time_stamp+'/reflectance/band_2'][:]/cosSZA
            obs = get_NDVI(Ref_red, Ref_NIR1).flatten()
        elif observable=='NDSI':
            Ref_grn  = MODIS_database[time_stamp+'/reflectance/band_4'][:]/cosSZA
            Ref_NIR2 = MODIS_database[time_stamp+'/reflectance/band_6'][:]/cosSZA
            obs = get_NDSI(Ref_grn, Ref_NIR2).flatten()
        elif observable=='SVI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA

            numcols, numrows = np.shape(Ref_red)[0], np.shape(Ref_red)[1]
            obs = get_spatial_variability_index(Ref_red,numcols, numrows).flatten()
        elif observable=='Cirrus':
            Ref_NIR3 = MODIS_database[time_stamp+'/reflectance/band_26'][:]/cosSZA

            obs = get_cirrus_Ref(Ref_NIR3).flatten()
        elif observable=='Vis_Ref':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA

            obs = get_visible_reflectance(Ref_red).flatten()
        else: #for NIR_Ref
            Ref_NIR1 = MODIS_database[time_stamp+'/reflectance/band_2'][:]/cosSZA

            obs = get_NIR_reflectance(Ref_NIR1).flatten()





        #find where the calculated OLP matches with the desired OLP
        #Here OLP has been flattened across axis 1 and 2
        np.where((OLP[:, 0] == observable_level_paramter) & \
                 (OLP[:, 1] == observable_level_paramter) & \
                 (OLP[:, 2] == observable_level_paramter) & \
                 (OLP[:, 3] == observable_level_paramter) & \
                 (OLP[:, 4] == observable_level_paramter) & \
                 (OLP[:, 5] == observable_level_paramter)   )

        #I'll need 6 sets of if statements
        #(cosSZA, VZA, RAZ, TA, scene_ID, DOY)
        #where the input will be the bin we're calculating the hist for
        #observable_level_parameter = [0,0,0,0,0,0]

        # idx = np.where()
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]



        # #use only entries in a particular cos(SZA) range
        # idx = np.where((cosSZA.flatten() >0) & (cosSZA.flatten() < 0.5))
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]
        #
        # #only land data points
        # idx = np.where(LWM.flatten() != 0)
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]

        # #only water no glint data points
        # idx = np.where((LWM.flatten() == 0) & (SGM.flatten() == 1))
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]

        # #only sunglint data points
        # idx = np.where(LWM.flatten() == 0 & (SGM.flatten() == 0))
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]

        # #only snow/ice data points
        # idx = np.where(SIM.flatten() == 0)
        # obs = obs[idx]
        # cloud_mask = cloud_mask[idx]

    obs_temp = obs[(cloud_mask == 1) & (obs > -5)]
    Threshold = np.percentile(obs_temp, 1)
    print('{} threhshold {:02.5f}'.format(observable, Threshold))

    return cloudy_count, clear_count, Threshold

def calc_threshold(observable, observable_level_paramter, target_area, sfc_ID, \
 file_path = '/Users/vllgsbr2/Desktop/MODIS_Training/Data/toronto_PTA_Subsets.HDF5'):
    '''
    Objective:
        Calculate the threhsolds based on clear data points based on a particular
        observable and observable level parameters.

    Arguments:
        observable {2D array} -- some observable
        observable_level_paramter {1D array} -- only use data points that fall
                                                fall wihtin these bins. In order
                                                they are:
                                                [cosSZA,VZA,RAZ,TA,Scene_ID,DOY]
        target_area {int} -- integer denoting what the target area is.
                             (no final list onto which target is what number)
        sfc_ID {2D narray} -- pass in the SMB sfc_ID map corresponding to the target
        sfcID_lat, sfcID_lon {2D narray} -- lat/lon of sfc_ID map
        file_path {str} -- path to database from create_database.py

    Return:
        Threshold {float} -- where 1% of clear data points misclassfied, as in
                             MOD35 visible reflectance test.
        cloudy_count, clear_count {int} -- number of MOD35 confident cloudy data
                                           points, all other classes of clear
    '''

    import numpy as np
    import h5py
    import sys
    sys.path.insert(0,'/Users/vllgsbr2/Desktop/MAIA_JPL_code/JPL_MAIA_CM')
    from JPL_MCM import get_whiteness_index, get_NDVI, get_NDSI, get_visible_reflectance,\
    get_NIR_reflectance, get_cirrus_Ref, get_sun_glint_mask, get_observable_level_parameter
    from svi_dynamic_size_input import svi_calculation


    MODIS_database = h5py.File(file_path, 'r')
    time_stamps = list(MODIS_database.keys())

    cloudy_count = 0
    clear_count  = 0

    for epoch, time_stamp in enumerate(time_stamps):
        #retrieve cloud mask for each pixel
        cloud_mask = MODIS_database[time_stamp+'/cloud_mask/Unobstructed_FOV_Quality_Flag'][:]
        #confident cloudy 0; all else 1
        cloud_mask[cloud_mask != 0] = 1
        cloud_mask = cloud_mask.flatten()

        cloudy_count  += len(np.where(cloud_mask==0)[0])
        clear_count   += len(np.where(cloud_mask==1)[0])
        cloud_fraction = (len(np.where(cloud_mask==0)[0])/cloud_mask.size)#*100


        #grab observable level parameters from MODIS
        #->(cosSZA, VZA, RAZ, TA, scene_ID, DOY)

        #grab land_water/sunglint/snow-ice mask
        LWM = MODIS_database[time_stamp+'/cloud_mask/Land_Water_Flag'][:]
        SIM = MODIS_database[time_stamp+'/cloud_mask/Snow_Ice_Background_Flag'][:]
        SGM = MODIS_database[time_stamp+'/cloud_mask/Sun_glint_Flag'][:]

        #grab obeservable level paramters
        SZA    = MODIS_database[time_stamp+'/sunView_geometry/solarZenith'][:]
        cosSZA = np.cos(np.deg2rad(SZA))
        VZA    = MODIS_database[time_stamp+'/sunView_geometry/sensorZenith'][:]
        VAA    = MODIS_database[time_stamp+'/sunView_geometry/sensorAzimuth'][:]
        SAA    = MODIS_database[time_stamp+'/sunView_geometry/solarAzimuth'][:]
        RAZ    = VAA - 180.0 - SAA
        TA     = target_area

        #grab MODIS lat/lon
        MOD03_lat = MODIS_database[time_stamp+'/geolocation/lat'][:]
        MOD03_lon = MODIS_database[time_stamp+'/geolocation/lon'][:]

        #calculate observable level parameter for the scence ID
        #Note this is the retrieved OLP to compare with OLP passed into this
        #function.

        Scene_ID = make_sceneID(LWM, SGM, SIM, sfc_ID)
        OLP = get_observable_level_parameter(SZA, VZA, SAA, VAA, TA, Scene_ID, DOY)

        #use output from regriding function to only selcect points valid for the
        #sfc_ID available

        #convert MODIS Reflectance into BRF
        #RGB; NIR_X - 1,2,3 = NDVI,NDSI,Cirrus
        if observable=='WI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA
            Ref_grn  = MODIS_database[time_stamp+'/reflectance/band_4'][:]/cosSZA
            Ref_blu  = MODIS_database[time_stamp+'/reflectance/band_3'][:]/cosSZA

            #flatten output to count instances of bins
            obs = get_whiteness_index(Ref_red, Ref_grn, Ref_blu).flatten()

        elif observable=='NDVI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA
            Ref_NIR1 = MODIS_database[time_stamp+'/reflectance/band_2'][:]/cosSZA
            obs = get_NDVI(Ref_red, Ref_NIR1).flatten()
        elif observable=='NDSI':
            Ref_grn  = MODIS_database[time_stamp+'/reflectance/band_4'][:]/cosSZA
            Ref_NIR2 = MODIS_database[time_stamp+'/reflectance/band_6'][:]/cosSZA
            obs = get_NDSI(Ref_grn, Ref_NIR2).flatten()
        elif observable=='SVI':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA

            numcols, numrows = np.shape(Ref_red)[0], np.shape(Ref_red)[1]
            obs = get_spatial_variability_index(Ref_red,numcols, numrows).flatten()
        elif observable=='Cirrus':
            Ref_NIR3 = MODIS_database[time_stamp+'/reflectance/band_26'][:]/cosSZA

            obs = get_cirrus_Ref(Ref_NIR3).flatten()
        elif observable=='Vis_Ref':
            Ref_red  = MODIS_database[time_stamp+'/reflectance/band_1'][:]/cosSZA

            obs = get_visible_reflectance(Ref_red).flatten()
        else: #for NIR_Ref
            Ref_NIR1 = MODIS_database[time_stamp+'/reflectance/band_2'][:]/cosSZA

            obs = get_NIR_reflectance(Ref_NIR1).flatten()





        #find where the calculated OLP matches with the desired OLP
        #Here OLP has been flattened across axis 1 and 2
        np.where((OLP[:, 0] == observable_level_paramter) & \
                 (OLP[:, 1] == observable_level_paramter) & \
                 (OLP[:, 2] == observable_level_paramter) & \
                 (OLP[:, 3] == observable_level_paramter) & \
                 (OLP[:, 4] == observable_level_paramter) & \
                 (OLP[:, 5] == observable_level_paramter)   )


    obs_temp = obs[(cloud_mask == 1) & (obs > -5)]
    Threshold = np.percentile(obs_temp, 1)
    print('{} threhshold {:02.5f}'.format(observable, Threshold))

    return cloudy_count, clear_count, Threshold

def load_thresh_database(hf_file, idx, thresholds):

    '''
    Objective:
        After calculating
    Arguments:
        hf_file {hdf5 data set object} -- Threshold data base file to load.
        thresholds {1D narray} -- Thresholds to store
        idx {2D narray (X, 6)} -- For every threshold, contains the index in
                                  which it should be stored in hf_dataset.
    Return:
        No return
    '''

    hf_file['Thresholds'][idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]] = \
                                                                    thresholds



if __name__ == '__main__':

    save_path = './'
    hf_path   = save_path + 'LA_threshold_database.HDF5'
    hf        = h5py.File(hf_path, 'w')
    LA_database_path = ''
    #create structure in hdf file to send into load_thresh_database
    group = hf.create_group('Thresholds')
    observable
    observable_level_paramter
    target_area
    sfc_ID
    thresholds, idx = calc_thresholds(observable, observable_level_paramter,\
                              target_area, sfc_ID, file_path = LA_database_path)
    load_thresh_database(hf, idx, thresholds)

    hf.close()
