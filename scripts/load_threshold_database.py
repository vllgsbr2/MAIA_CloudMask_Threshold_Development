import numpy as np
import h5py
import mpi4py.MPI as MPI
from calc_observables import get_R, get_sun_glint_mask, get_whiteness_index,\
get_NDVI, get_NDSI, get_visible_reflectance, get_NIR_reflectance,\
get_spatial_variability_index, get_cirrus_Ref


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
def new_thresh_file(hf_path):
    hf    = h5py.File(hf_path, 'w')

    #create structure in hdf file
    group = hf.create_group('Thresholds')

    #save empty thresholds to sub group
    #(cos_SZA, VZA, RAZ, Target Area, scene_type_identifier, DOY)
    #where test isnt applied fill value will be used
    #(10,14,12,30,33,46) (1,1,1,1,1,1)
    threshold_verifi = {'realistic': [0.03, 0.01, 0.15, 0.03, 0.15, 0.1, 0.2]}
    file_type = 'realistic'

    T_NDVI    = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][0]
    T_NDSI    = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][1]
    T_SVI     = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][2]
    T_WI      = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][3]
    T_Cirrus  = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][4]
    T_NIR_Ref = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][5]
    T_VIS_Ref = np.ones((10,14,12,30,33,46)) * threshold_verifi[file_type][6]

    group.create_dataset('T_NDVI'   , data=T_NDVI   , dtype='f', compression='gzip')
    group.create_dataset('T_NDSI'   , data=T_NDSI   , dtype='f', compression='gzip')
    group.create_dataset('T_SVI'    , data=T_SVI    , dtype='f', compression='gzip')
    group.create_dataset('T_WI'     , data=T_WI     , dtype='f', compression='gzip')
    group.create_dataset('T_Cirrus' , data=T_Cirrus , dtype='f', compression='gzip')
    group.create_dataset('T_NIR_Ref', data=T_NIR_Ref, dtype='f', compression='gzip')
    group.create_dataset('T_VIS_Ref', data=T_VIS_Ref, dtype='f', compression='gzip')

    #add attributes to subgroups to label the dimensions
    labels      = ['cos_SZA', 'VZA', 'RAZ', 'Target_Area',\
                    'surface_ID', 'DOY']
    observables = ['T_NDVI', 'T_NDSI', 'T_SVI', 'T_WI', 'T_Cirrus', 'T_NIR_Ref',\
                   'T_VIS_Ref']

    for obs in observables:
        for i, label in enumerate(labels):
            hf['Thresholds/' + obs].dims[i].label = label

    return hf

if __name__ == '__main__':

    # #initialize mpi insatnce
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    save_path = '../thresholds'
    hf_path   = save_path + 'LA_threshold_database.HDF5'

    try:
        hf = new_thresh_file(hf_path)
    except:
        hf = h5py.File(hf_path, 'r+')

    #calculate thresholds
    LA_database_path = '' #this is where the 16 years of data is stored over LA
    thresholds, idx = calc_thresholds(observable, observable_level_paramter,\
                              target_area, sfc_ID, file_path = LA_database_path)
                              
    #load thresholds into hdf file
    load_thresh_database(hf, idx, thresholds)

    hf.close()
