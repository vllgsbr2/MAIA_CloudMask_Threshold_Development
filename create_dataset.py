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

#import matplotlib.pyplot as plt

def save_crop(subgroup, dataset_name, cropped_data):
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
    #add dataset to 'group'
    subgroup.create_dataset(dataset_name, data=cropped_data, compression="gzip")

def build_data_base(filename_MOD_02, filename_MOD_03, filename_MOD_35, hf_path, hf, \
                    group_name, fieldname, target_lat, target_lon, col_mesh, row_mesh):
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

    t0 = time.time()

    rad_or_ref              = True
    radiance_250_Aggr1km    = prepare_data(filename_MOD_02, fieldname[1], rad_or_ref)
    radiance_500_Aggr1km    = prepare_data(filename_MOD_02, fieldname[3], rad_or_ref)
    radiance_1KM            = prepare_data(filename_MOD_02, fieldname[4], rad_or_ref)

    rad_or_ref              = False
    reflectance_250_Aggr1km = prepare_data(filename_MOD_02, fieldname[1], rad_or_ref)
    reflectance_500_Aggr1km = prepare_data(filename_MOD_02, fieldname[3], rad_or_ref)
    reflectance_1KM         = prepare_data(filename_MOD_02, fieldname[4], rad_or_ref)

    #calculate geolocation
    lat = get_lat(filename_MOD_03).astype(np.float64)
    lon = get_lon(filename_MOD_03).astype(np.float64)

    #calculate geometry
    solarZenith   = get_solarZenith(filename_MOD_03)
    sensorZenith  = get_sensorZenith(filename_MOD_03)
    solarAzimuth  = get_solarAzimuth(filename_MOD_03)
    sensorAzimuth = get_sensorAzimuth(filename_MOD_03)


    #calculate cloudmask
    data_SD           = get_data(filename_MOD_35, 'Cloud_Mask', 2)
    data_SD_bit       = get_bits(data_SD, 0)
    data_decoded_bits = decode_byte_1(data_SD_bit)

    #calculate cloud mask tests
    data_SD_cloud_mask = data_SD
    decoded_cloud_mask_tests = decode_tests(data_SD_cloud_mask, filename_MOD_35)

    #ceate structure in hdf file
    group                       = hf.create_group(group_name)
    subgroup_radiance           = group.create_group('radiance')
    subgroup_reflectance        = group.create_group('reflectance')
    subgroup_geolocation        = group.create_group('geolocation')
    subgroup_sunView_geometry   = group.create_group('sunView_geometry')
    subgroup_cloud_mask         = group.create_group('cloud_mask')
    subgroup_cloud_mask_test    = group.create_group('cloud_mask_tests')

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

    #import matplotlib.pyplot as plt
    #f, ax = plt.subplots(ncols=2)
    #ax[0].imshow(regrid_row_idx)
    #ax[1].imshow(regrid_col_idx)
    #plt.show()

    #crop and save the datasets*************************************************

    #reflectance and radiance
    band_index = {'1':0,
                  '2':1,
                  '3':0,
                  '4':1,
                  '6':3,
                  '8':0,
                  '12':4,
                  '26':12
                  }

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

        # if band=='1':
        #     f, ax = plt.subplots(ncols=2)
        #
        #     reflectance_250_Aggr1km[index][regrid_row_idx, regrid_col_idx]+= 0.2
        #
        #     ax[0].imshow(reflectance_250_Aggr1km[index])
        #     ax[1].imshow(crop_reflectance)
        #     ax[0].set_title('ref band 1')

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

    # f1, ax1 = plt.subplots(ncols=2)
    # sun_val[regrid_row_idx, regrid_col_idx]+= 2
    # ax1[0].imshow(sun_val)
    # ax1[1].imshow(crop_sun)
    # ax1[0].set_title('SZA')

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

    # f2, ax2 = plt.subplots(ncols=2)
    # cm_val[regrid_row_idx, regrid_col_idx]+= 2
    # ax2[0].imshow(cm_val)
    # ax2[1].imshow(crop_cm)
    # ax2[0].set_title('Cloud Mask')

    #*******************************************************************************
    #cloud mask tests
    #cm test vals set to 9 are bad data
    cloud_mask_tests = {'High_Cloud_Flag_1380nm':decoded_cloud_mask_tests[0],\
                        'Cloud_Flag_Visible_Ratio':decoded_cloud_mask_tests[2],\
                        'Near_IR_Reflectance':decoded_cloud_mask_tests[3],\
                        'Cloud_Flag_Spatial_Variability':decoded_cloud_mask_tests[4],\
                        'Cloud_Flag_Visible_Reflectance':decoded_cloud_mask_tests[1]\
                        }
    for cm_test_key, cm_test_val in cloud_mask_tests.items():
        cm_test_val  = cm_test_val.astype(np.float64)
        crop_cm_test = cm_test_val[regrid_row_idx, regrid_col_idx]

        #Apply fill values
        crop_cm_test[fill_val_idx] = fill_val

        save_crop(subgroup_cloud_mask_test, cm_test_key, crop_cm_test)

    # f3, ax3 = plt.subplots(ncols=2)
    # cm_test_val[regrid_row_idx, regrid_col_idx]+= 2
    # ax3[0].imshow(cm_test_val)
    # ax3[1].imshow(crop_cm_test)
    # ax3[0].set_title('Cloud Mask tests')


    # plt.show()

if __name__ == '__main__':

    import pandas as pd
    import tables
    tables.file._open_files.close_all()
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
    filename_MOD_03_timeStamp = [x[7:19]  for x in filename_MOD_03]
    filename_MOD_35_timeStamp = [x[10:22] for x in filename_MOD_35]

    #add full path to file
    filename_MOD_02 = [PTA_file_path + '/MOD_02/' + x for x in filename_MOD_02]
    filename_MOD_03 = [PTA_file_path + '/MOD_03/' + x for x in filename_MOD_03]
    filename_MOD_35 = [PTA_file_path + '/MOD_35/' + x for x in filename_MOD_35]

    #create/open file
    file_num = sys.argv[3]
    hf_path = PTA_file_path + '/LA_PTA_database_'+file_num+'.hdf5'
    hf      = h5py.File(hf_path, 'w')

    #initialize some constants outside of the loop
    fieldname       = ['EV_250_RefSB', 'EV_250_Aggr1km_RefSB',\
                       'EV_500_RefSB', 'EV_500_Aggr1km_RefSB',\
                       'EV_1KM_RefSB']

    file_MAIA  = '/data/keeling/a/vllgsbr2/c/LA_PTA_MAIA.hdf5'
    file_MAIA  = h5py.File(file_MAIA, 'r')
    target_lat = file_MAIA['lat'][()].astype(np.float64)
    target_lon = file_MAIA['lon'][()].astype(np.float64)

    #new indices to regrid, use that as numpy where if you will
    nx, ny = (2030, 1354)
    rows = np.arange(2030)
    cols = np.arange(1354)
    col_mesh, row_mesh = np.meshgrid(cols, rows)

    t0 = time.time()

    #open file to write status of algorithm to
    output = open('create_dataset_status.txt', 'w+')

    i=1
    start = 0
    end   = len(filename_MOD_02)
    print(end)
    start, end = int(sys.argv[1]), int(sys.argv[2])
    for MOD02, MOD03, MOD35, time_MOD02, time_MOD03, time_MOD35\
                            in zip(filename_MOD_02[start:end]          ,\
                                   filename_MOD_03[start:end]          ,\
                                   filename_MOD_35[start:end]          ,\
                                   filename_MOD_02_timeStamp[start:end],\
                                   filename_MOD_03_timeStamp[start:end],\
                                   filename_MOD_35_timeStamp[start:end]):

        #home       = '/data/keeling/a/vllgsbr2/c/LA_test_case_data/'
        #MOD03 = home + 'MOD03.A2017246.1855.061.2017257170030.hdf'
        #MOD02 = home + 'MOD021KM.A2017246.1855.061.2017258202757.hdf'
        try:
            build_data_base(MOD02, MOD03, MOD35, hf_path, hf, time_MOD02, fieldname,\
                        target_lat, target_lon, col_mesh, row_mesh)

            output.write('{:0>5d} {} {}'.format(i, time_MOD02, 'added to database\n'))
        except Exception as e:
            output.write('{:0>5d} {} {} {}'.format(i, time_MOD02, e, '\n'))
        i+=1

    hf.close()
    output.write('{:2.2f}'.format(time.time()-t0))
    output.close()

