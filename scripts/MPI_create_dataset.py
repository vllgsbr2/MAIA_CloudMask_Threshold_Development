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

def build_data_base(filename_MOD_02, filename_MOD_03, filename_MOD_35, hf_path, hf, \
                    group_name, fieldname, target_lat, target_lon):
    '''
    INPUT:
        filename_MOD_02 - str   - filepath to MOD02
        filename_MOD_03 - str   - filepath to MOD03
        filename_MOD_35 - str   - filepath to MOD35
        PTA_lat         - float - lat of projected target area
        PTA_lon         - float - lon of projected target area
        hf_path         - str   - file path to previously created hdf5 file to
                                  store it in. (1 HDF5 file)/PTA
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

    #calculate cloud mask tests
    data_SD_cloud_mask       = data_SD
    decoded_cloud_mask_tests = decode_tests(data_SD_cloud_mask, filename_MOD_35)

    #grab earth sun distance
    earth_sun_dist = get_earth_sun_dist(filename_MOD_02)

    #get MOD03 surface types
    MOD03_LandSeaMask = get_LandSeaMask(filename_MOD_03)

    hdf_file.end()

    #ceate structure in hdf file
    group                       = hf.create_group(group_name)
    subgroup_radiance           = group.create_group('radiance')
    #subgroup_reflectance        = group.create_group('reflectance')
    subgroup_scale_factors      = group.create_group('scale_factors')
    subgroup_geolocation        = group.create_group('geolocation')
    subgroup_sunView_geometry   = group.create_group('sunView_geometry')
    subgroup_cloud_mask         = group.create_group('cloud_mask')
    #subgroup_cloud_mask_test    = group.create_group('cloud_mask_tests')


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

    #grab -999 fill values in regrid col/row idx
    #use these positions to write fill values when regridding the rest of the data
    fill_val = -999
    fill_val_idx = np.where((regrid_row_idx == fill_val) | \
                            (regrid_col_idx == fill_val)   )

    regrid_row_idx[fill_val_idx] = regrid_row_idx[0,0]
    regrid_col_idx[fill_val_idx] = regrid_col_idx[0,0]

    #crop and save the datasets*************************************************

    #save band weighted solar irradiance
    save_crop(group, 'band_weighted_solar_irradiance', E_std_0)

    #save earth sun distance
    save_crop(group, 'earth_sun_distance', earth_sun_dist, compress=False)

    #reflectance and radiance

    for band, index in band_index.items():
        if band=='1' or band=='2':
            crop_radiance = radiance_250_Aggr1km[index][regrid_row_idx, regrid_col_idx]
            #crop_reflectance = reflectance_250_Aggr1km[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            #crop_reflectance[fill_val_idx] = fill_val

        elif band=='3' or band=='4' or band=='6':
            crop_radiance = radiance_500_Aggr1km[index][regrid_row_idx, regrid_col_idx]
            #crop_reflectance = reflectance_500_Aggr1km[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            #crop_reflectance[fill_val_idx] = fill_val

        else:
            crop_radiance = radiance_1KM[index][regrid_row_idx, regrid_col_idx]
            #crop_reflectance = reflectance_1KM[index][regrid_row_idx, regrid_col_idx]

            #Apply fill values
            crop_radiance[fill_val_idx]    = fill_val
            #crop_reflectance[fill_val_idx] = fill_val

        #group_name is granule, radiance is subgroup, band_1 is dataset, then the data
        save_crop(subgroup_radiance, 'band_{}'.format(band), crop_radiance)
        #save_crop(subgroup_reflectance, 'band_{}'.format(band), crop_reflectance)

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
    cloud_mask = {'Cloud_Mask_Flag':data_decoded_bits[0],\
                  'Day_Night_Flag':data_decoded_bits[2],\
                  'Sun_glint_Flag':data_decoded_bits[3],\
                  'Snow_Ice_Background_Flag':data_decoded_bits[4],\
                  'Land_Water_Flag':data_decoded_bits[5],\
                  'Unobstructed_FOV_Quality_Flag':data_decoded_bits[1]\
                  }
    for cm_key, cm_val in cloud_mask.items():
        crop_cm = cm_val[regrid_row_idx, regrid_col_idx]

        #Apply fill values
        crop_cm[fill_val_idx] = fill_val

        save_crop(subgroup_cloud_mask, cm_key, crop_cm)

    #*******************************************************************************
    #add in MOD03 surface types
    crop_MOD03_LandSeaMask = MOD03_LandSeaMask[regrid_row_idx, regrid_col_idx]
    #put in main group since it is not compatible in a sub group
    save_crop(group, 'MOD03_LandSeaMask', crop_MOD03_LandSeaMask)

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
    import tables
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
            config = configparser.ConfigParser()
            config.read(config_home_path+'/test_config.txt')

            home     = config['home']['home']
            PTA      = config['current PTA']['PTA']
            PTA_path = config['PTAs'][PTA]

            MODXX      = config['supporting directories']['MODXX']
            MOD02_path = '{}/{}/{}{}'.format(home, PTA_path, MODXX, '02')
            MOD03_path = '{}/{}/{}{}'.format(home, PTA_path, MODXX, '03')
            MOD35_path = '{}/{}/{}{}'.format(home, PTA_path, MODXX, '35')

            #grab files names for PTA and sort them
            filename_MOD_02 = np.sort(np.array(os.listdir(MOD02_path)))
            filename_MOD_03 = np.sort(np.array(os.listdir(MOD03_path)))
            filename_MOD_35 = np.sort(np.array(os.listdir(MOD35_path)))

            #grab time stamp (YYYYDDD.HHMM) to name each group after the granule
            #it comes from
            filename_MOD_02_timeStamp = [x[10:22] for x in filename_MOD_02]

            #add full path to file
            filename_MOD_02 = [MOD02_path + '/' + x for x in filename_MOD_02]
            filename_MOD_03 = [MOD03_path + '/' + x for x in filename_MOD_03]
            filename_MOD_35 = [MOD35_path + '/' + x for x in filename_MOD_35]

            #initialize global constants outside of the loop
            fieldname = ['EV_250_RefSB', 'EV_250_Aggr1km_RefSB',\
                         'EV_500_RefSB', 'EV_500_Aggr1km_RefSB',\
                         'EV_1KM_RefSB']

            #grab target lat/lon from Guangyu h5 files (new JPL grids)
            PTA_grid_file_path = config['PTA lat/lon grid files'][PTA]
            filepath_latlon = '{}/{}'.format(home, PTA_grid_file_path)
            with h5py.File(filepath_latlon, 'r') as hf_latlon:
                target_lat = hf_latlon['Geolocation/Latitude'][()].astype(np.float64)
                target_lon = hf_latlon['Geolocation/Longitude'][()].astype(np.float64)

            #define start and end file for a particular rank
            #(size - 1) so last processesor can take the modulus
            end               = len(filename_MOD_02)
            processes_per_cpu = end // (size-1)
            start             = rank * processes_per_cpu

            if rank < (size-1):
                end = (rank+1) * processes_per_cpu
            elif rank==(size-1):
                processes_per_cpu_last = end % (size-1)
                end = (rank * processes_per_cpu) + processes_per_cpu_last

            #create/open file
            #open file to write status of algorithm to
            database_loc = '{}/{}/{}'.format(home, PTA_path, config['supporting directories']['Database'])
            hf_path = '{}/{}_PTA_database_rank_{:02d}.hdf5'.format(database_loc, PTA, rank)
            output_path = '{}/{}/Database_Diagnostics/diagnostics_{:02d}.txt'.format(home, PTA_path, rank)

            with h5py.File(hf_path, 'w') as hf, open(output_path, 'w') as output:
                i=1

                for MOD02, MOD03, MOD35, time_MOD02\
                                   in zip(filename_MOD_02[start:end]          ,\
                                          filename_MOD_03[start:end]          ,\
                                          filename_MOD_35[start:end]          ,\
                                          filename_MOD_02_timeStamp[start:end]):
                    # print('{}\n{}\n{}\n{}\n{}'.format(i, MOD02, MOD03, MOD35, time_MOD02))

                    try:
                        build_data_base(MOD02, MOD03, MOD35, hf_path, hf,\
                                    time_MOD02, fieldname, target_lat,\
                                    target_lon)

                        output.write('{:0>5d}, {}, {}'.format(i, time_MOD02, 'added to database\n'))
                        print(i, time_MOD02, '\n')
                    except Exception as e:

                        output.write('{:0>5d}, {}, {}, {}'.format(i, time_MOD02, e, 'corrupt\n'))
                        print(i, time_MOD02, 'corrupt\n')
                    i+=1
