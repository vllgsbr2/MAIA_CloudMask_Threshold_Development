'''
author: Javier Villegas
Format data base into hdf5 file containing every time stamp associated with a
granule, with associated dataset of radiance, reflectance, cloudmask, sun view
geometry, and geolocation.
'''
from read_MODIS_02 import prepare_data
from read_MODIS_03 import *
from read_MODIS_35 import *
from regrid import regrid_MODIS_2_MAIA
import h5py
import sys
import os
import time
import pytaf
from pyhdf.SD import SD

#import matplotlib.pyplot as plt

def save_crop(subgroup, dataset_name, cropped_data, compress=True):
    '''
    INPUT:
          cropped data from regrid_MODIS_2_MAIA
          subgroup: - h5py group - belongs to subgroup of MODIS dataset i.e.
                                   radiance
          dataset_name: - str - name to save data to
    RETURN:
          save cropped data into hdf5 file without closing it. This is to add
          data into the hdf5 file and then freeing dereferenced pointers in
          order to have good memory management
    '''
    try:
        if compress:
            #add dataset to 'group'
            subgroup.create_dataset(dataset_name, data=cropped_data, compression="gzip")
        else:
            subgroup.create_dataset(dataset_name, data=cropped_data)
    except:
        subgroup[dataset_name][:] = cropped_data

def build_data_base(filename_MOD_02, filename_MOD_03, filename_MOD_35, hf, \
                    group_name, fieldname, target_lat, target_lon):
    '''
    INPUT:
        filename_MOD_02 - str   - filepath to MOD02
        filename_MOD_03 - str   - filepath to MOD03
        filename_MOD_35 - str   - filepath to MOD35
        PTA_lat         - float - lat of projected target area
        PTA_lon         - float - lon of projected target area
        hf              - h5py file object - file object to database file
        group_name      - str   - time stamp of granule to name subgroup
    RETURN:
        saves all calculated fields into hdf 5 file structure
    '''

    rad_or_ref              = True
    radiance_250_Aggr1km, scale_factor_rad_250m, scale_factor_ref_250m    = prepare_data(filename_MOD_02, fieldname[1], rad_or_ref)
    radiance_500_Aggr1km, scale_factor_rad_500m, scale_factor_ref_500m    = prepare_data(filename_MOD_02, fieldname[3], rad_or_ref)
    radiance_1KM, scale_factor_rad_1km, scale_factor_ref_1km              = prepare_data(filename_MOD_02, fieldname[4], rad_or_ref)

    rad_or_ref              = False
    reflectance_250_Aggr1km, scale_factor_rad_250m, scale_factor_ref_250m = prepare_data(filename_MOD_02, fieldname[1], rad_or_ref)
    reflectance_500_Aggr1km, scale_factor_rad_500m, scale_factor_ref_500m = prepare_data(filename_MOD_02, fieldname[3], rad_or_ref)
    reflectance_1KM, scale_factor_rad_1km, scale_factor_ref_1km           = prepare_data(filename_MOD_02, fieldname[4], rad_or_ref)

    #grab scale factors for MODIS bands 3,4,1,2,6,26 (MAIA bands 4,5,6,9,12,13)
    band_index = {'1':0,
                  '2':1,
                  '3':0,
                  '4':1,
                  '6':3,
                  '26':14
                  }

    scale_factor_ref_500m_final = []
    scale_factor_rad_500m_final = []
    scale_factor_ref_1km_final  = []
    scale_factor_rad_1km_final  = []

    for band, index in band_index.items():
        band = int(band)

        if band == 3 or band == 4 or band ==6:
            scale_factor_ref_500m_final.append(scale_factor_ref_500m[index])
            scale_factor_rad_500m_final.append(scale_factor_rad_500m[index])
        elif band == 26:
            scale_factor_ref_1km_final.append(scale_factor_ref_1km[index])
            scale_factor_rad_1km_final.append(scale_factor_rad_1km[index])
        else:
            pass
    scale_factor_ref_500m_final = np.array(scale_factor_ref_500m_final)
    scale_factor_rad_500m_final = np.array(scale_factor_rad_500m_final)
    scale_factor_ref_1km_final  = np.array(scale_factor_ref_1km_final)
    scale_factor_rad_1km_final  = np.array(scale_factor_rad_1km_final)

    #in order MAIA  bands 6,9,4,5,12,13
    #in order MODIS bands 1,2,3,4,6 ,26
    scale_factor_rad = np.concatenate((scale_factor_rad_250m, scale_factor_rad_500m_final, scale_factor_rad_1km_final), axis=0)
    scale_factor_ref = np.concatenate((scale_factor_ref_250m, scale_factor_ref_500m_final, scale_factor_ref_1km_final), axis=0)

    #calculate band weighted solar irradiance using scale factors above
    E_std_0 = np.pi * scale_factor_rad / scale_factor_ref

    #calculate geolocation
    lat = get_lat(filename_MOD_03).astype(np.float64)
    lon = get_lon(filename_MOD_03).astype(np.float64)

    #calculate geometry
    solarZenith   = get_solarZenith(filename_MOD_03)
    sensorZenith  = get_sensorZenith(filename_MOD_03)
    solarAzimuth  = get_solarAzimuth(filename_MOD_03)
    sensorAzimuth = get_sensorAzimuth(filename_MOD_03)

    #calculate cloudmask
    data_SD, hdf_file = get_data(filename_MOD_35, 'Cloud_Mask', 2, True)
    data_SD_bit       = get_bits(data_SD, 0)
    data_decoded_bits = decode_byte_1(data_SD_bit)
    cloud_mask        = np.copy(data_decoded_bits[1])


    #calculate cloud mask tests
    data_SD_cloud_mask       = data_SD
    decoded_cloud_mask_tests = decode_tests(data_SD_cloud_mask, filename_MOD_35)
    #only take the overall qaulity flag; 0 not determined, 1 determined, 2 not good QA
    cloud_mask_QA = np.copy(decoded_cloud_mask_tests[-1])
    #mask the cloud_mask with QA fill vals for downstream use
    cloud_mask[cloud_mask_QA != 1] = -998
    quality_screened_cloud_mask = cloud_mask

    #grab earth sun distance
    earth_sun_dist = get_earth_sun_dist(filename_MOD_02)

    hdf_file.end()

    nx, ny = np.shape(lat)
    rows = np.arange(nx)
    cols = np.arange(ny)
    col_mesh, row_mesh = np.meshgrid(cols, rows)

    regrid_row_idx = regrid_MODIS_2_MAIA(np.copy(lat),\
                                         np.copy(lon),\
                                         np.copy(target_lat),\
                                         np.copy(target_lon),\
                                         np.copy(row_mesh).astype(np.float64)).astype(np.int)

    regrid_col_idx = regrid_MODIS_2_MAIA(np.copy(lat),\
                                         np.copy(lon),\
                                         np.copy(target_lat),\
                                         np.copy(target_lon),\
                                         np.copy(col_mesh).astype(np.float64)).astype(np.int)

    #if no data was regridded skip recording the scene
    if (regrid_row_idx[regrid_row_idx >=0].size <= 0) | \
       (regrid_col_idx[regrid_col_idx >=0].size <= 0    ):
        return
    #grab -999 fill values in regrid col/row idx
    #use these positions to write fill values when regridding the rest of the data
    fill_val = -999
    fill_val_idx = np.where((regrid_row_idx < 0) | \
                            (regrid_col_idx < 0)   )

    regrid_row_idx[fill_val_idx] = regrid_row_idx[regrid_row_idx >= 0][0]
    regrid_col_idx[fill_val_idx] = regrid_col_idx[regrid_col_idx >= 0][0]

    #crop and save the datasets*************************************************

    #ceate structure in hdf file
    group                       = hf.create_group(group_name)
    subgroup_radiance           = group.create_group('radiance')
    subgroup_reflectance           = group.create_group('reflectance')
    # subgroup_scale_factors      = group.create_group('scale_factors')
    subgroup_geolocation        = group.create_group('geolocation')
    subgroup_sunView_geometry   = group.create_group('sunView_geometry')
    subgroup_cloud_mask         = group.create_group('cloud_mask')
    #subgroup_cloud_mask_test    = group.create_group('cloud_mask_tests')

    #save band weighted solar irradiance
    save_crop(group, 'band_weighted_solar_irradiance', E_std_0)

    #save earth sun distance
    save_crop(group, 'earth_sun_distance', earth_sun_dist, compress=False)

    #reflectance and radiance; dont calc reflectance

    for band, index in band_index.items():
        if band=='1' or band=='2':
            crop_radiance = radiance_250_Aggr1km[index][regrid_row_idx, regrid_col_idx]
            crop_reflectance = reflectance_250_Aggr1km[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            crop_reflectance[fill_val_idx] = fill_val

        elif band=='3' or band=='4' or band=='6':
            crop_radiance = radiance_500_Aggr1km[index][regrid_row_idx, regrid_col_idx]
            crop_reflectance = reflectance_500_Aggr1km[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            crop_reflectance[fill_val_idx] = fill_val

        else:
            crop_radiance = radiance_1KM[index][regrid_row_idx, regrid_col_idx]
            crop_reflectance = reflectance_1KM[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            crop_reflectance[fill_val_idx] = fill_val

        #group_name is granule, radiance is subgroup, band_1 is dataset, then the data
        save_crop(subgroup_radiance, 'band_{}'.format(band), crop_radiance)
        save_crop(subgroup_reflectance, 'band_{}'.format(band), crop_reflectance)

    #*******************************************************************************
    #Sun view geometry
    sunView_geometry = {'solarAzimuth':solarAzimuth,\
                        'sensorAzimuth':sensorAzimuth,\
                        'solarZenith':solarZenith,\
                        'sensorZenith':sensorZenith
                        }
    for sun_key, sun_val in sunView_geometry.items():
        crop_sun = sun_val[regrid_row_idx, regrid_col_idx]

        #Apply fill values
        crop_sun[fill_val_idx] = fill_val

        save_crop(subgroup_sunView_geometry, sun_key, crop_sun)

    #*******************************************************************************
    #Geo Location
    crop_lat = np.copy(target_lat)
    crop_lon = np.copy(target_lon)

    #Apply fill values
    crop_lat[fill_val_idx] = fill_val
    crop_lon[fill_val_idx] = fill_val

    save_crop(subgroup_geolocation, 'lat', crop_lat)
    save_crop(subgroup_geolocation, 'lon', crop_lon)

    #*******************************************************************************
    #cloud mask
    cloud_mask = {'Cloud_Mask_Flag'              :data_decoded_bits[0],\
                  'Day_Night_Flag'               :data_decoded_bits[2],\
                  'Sun_glint_Flag'               :data_decoded_bits[3],\
                  'Snow_Ice_Background_Flag'     :data_decoded_bits[4],\
                  'Land_Water_Flag'              :data_decoded_bits[5],\
                  'Unobstructed_FOV_Quality_Flag':data_decoded_bits[1],\
                  'quality_screened_cloud_mask'  :quality_screened_cloud_mask\
                  }
    for cm_key, cm_val in cloud_mask.items():
        crop_cm = cm_val[regrid_row_idx, regrid_col_idx]

        #Apply fill values
        crop_cm[fill_val_idx] = fill_val

        save_crop(subgroup_cloud_mask, cm_key, crop_cm)

    #*******************************************************************************

    #cloud mask tests
    #cm test vals set to 9 are bad data
    #cloud_mask_tests = {'High_Cloud_Flag_1380nm':decoded_cloud_mask_tests[0],\
    #                    'Cloud_Flag_Visible_Ratio':decoded_cloud_mask_tests[2],\
    #                    'Near_IR_Reflectance':decoded_cloud_mask_tests[3],\
    #                    'Cloud_Flag_Spatial_Variability':decoded_cloud_mask_tests[4],\
    #                    'Cloud_Flag_Visible_Reflectance':decoded_cloud_mask_tests[1]\
    #                    }
    #for cm_test_key, cm_test_val in cloud_mask_tests.items():
    #    cm_test_val  = cm_test_val.astype(np.float64)
    #    crop_cm_test = cm_test_val[regrid_row_idx, regrid_col_idx]

    #    #Apply fill values
    #    crop_cm_test[fill_val_idx] = fill_val

    #    save_crop(subgroup_cloud_mask_test, cm_test_key, crop_cm_test)

if __name__ == '__main__':
    import mpi4py.MPI as MPI
    import configparser
    from get_MOD_02_03_35_paths import get_MODIS_file_paths, get_MODIS_file_paths_no_list
    from distribute_cores import distribute_processes

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
            config = configparser.ConfigParser()
            config.read(config_home_path+'/test_config.txt')

            home     = config['home']
            PTA      = config['current PTA']['PTA']
            PTA_path = home + config['PTAs'][PTA]

            if PTA == 'LosAngeles':
                MODXX      = config['supporting directories']['MODXX']
                MOD02_path = '{}/{}{}'.format(PTA_path, MODXX, '02')
                MOD03_path = '{}/{}{}'.format(PTA_path, MODXX, '03')
                MOD35_path = '{}/{}{}'.format(PTA_path, MODXX, '35')

                #grab files names for PTA and sort them
                filename_MOD_02 = np.sort(np.array(os.listdir(MOD02_path)))
                filename_MOD_03 = np.sort(np.array(os.listdir(MOD03_path)))
                filename_MOD_35 = np.sort(np.array(os.listdir(MOD35_path)))

                #add full path to file
                filename_MOD_02 = [MOD02_path + '/' + x for x in filename_MOD_02]
                filename_MOD_03 = [MOD03_path + '/' + x for x in filename_MOD_03]
                filename_MOD_35 = [MOD35_path + '/' + x for x in filename_MOD_35]

                filename_MOD_02,\
                filename_MOD_03,\
                filename_MOD_35 = get_MODIS_file_paths_no_list(filename_MOD_02,\
                                             filename_MOD_03, filename_MOD_35)

                filename_MOD_02_timeStamp = [x[-34:-22] for x in filename_MOD_02]

            else:

                MODXX_home = PTA_path + '/' + config['supporting directories']['MODXX_paths_lists']
                MOD02_txt = MODXX_home + '/MOD02_paths.txt'
                MOD03_txt = MODXX_home + '/MOD03_paths.txt'
                MOD35_txt = MODXX_home + '/MOD35_paths.txt'

                filename_MOD_02,\
                filename_MOD_03,\
                filename_MOD_35 = get_MODIS_file_paths(MOD02_txt, MOD03_txt, MOD35_txt)

                #time stamp is relative to MOD021KM standard file name
                #notice it counts from the end
                filename_MOD_02_timeStamp = [x[-34:-22] for x in filename_MOD_02]


            #initialize global constants outside of the loop
            fieldname = ['EV_250_RefSB', 'EV_250_Aggr1km_RefSB',\
                         'EV_500_RefSB', 'EV_500_Aggr1km_RefSB',\
                         'EV_1KM_RefSB']

            #grab target lat/lon from Guangyu h5 files (new JPL grids)
            PTA_grid_file_path = home + config['PTA lat/lon grid files'][PTA]

            with h5py.File(PTA_grid_file_path, 'r') as hf_latlon:
                target_lat = hf_latlon['Geolocation/Latitude'][()].astype(np.float64)
                target_lon = hf_latlon['Geolocation/Longitude'][()].astype(np.float64)

            #assign subset of files to current rank
            num_processes = len(filename_MOD_02)
            start, stop   = distribute_processes(size, num_processes)
            start, stop   = start[rank], stop[rank]

            #create/open file
            #open file to write status of algorithm to

            #base path for database file
            database_loc = '{}/{}'.format(PTA_path, config['supporting directories']['Database'])
            #complete database file path
            #add in a foler to make a seperate databse that doesn eleiminate empty granules
            # hf_path = '{}/{}_PTA_database_rank_{:02d}.hdf5'.format(database_loc, PTA, rank)
            hf_path = '{}/{}_PTA_database_rank_{:02d}.hdf5'.format(database_loc, PTA, rank)
            #diagnostic file path
            # output_path = '{}/Database_Diagnostics/diagnostics_{:02d}.txt'.format(PTA_path, rank)

            with h5py.File(hf_path, 'w') as hf:#, open(output_path, 'w') as output:
                i=1

                for MOD02, MOD03, MOD35, time_MOD02\
                                   in zip(filename_MOD_02[start:stop]          ,\
                                          filename_MOD_03[start:stop]          ,\
                                          filename_MOD_35[start:stop]          ,\
                                          filename_MOD_02_timeStamp[start:stop]):
                    # print('{}\n{}\n{}\n{}\n{}'.format(i, MOD02, MOD03, MOD35, time_MOD02))

                    try:
                        build_data_base(MOD02, MOD03, MOD35, hf,\
                                    time_MOD02, fieldname, target_lat,\
                                    target_lon)


                        # output.write('{:0>5d}, {}, {}'.format(i, time_MOD02, 'added to database\n'))
                        print(i, time_MOD02, '\n')
                    except Exception as e:

                        # output.write('{:0>5d}, {}, {}, {}'.format(i, time_MOD02, e, 'corrupt\n'))
                        print(i, time_MOD02, 'corrupt\n')
                    i+=1
