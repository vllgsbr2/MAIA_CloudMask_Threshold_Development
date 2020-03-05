def make_output(Sun_glint_exclusion_angle,\
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
                save_path=''):
    '''
    INPUTS:
          Sun_glint_exclusion_angle          -  float -
          Max_RDQI                           -  float -
          Max_valid_DTT, Min_valid_DTT       -  float -
          fill_val_1, fill_val_2, fill_val_3 -  float -
          Min_num_of_activated_tests         -  int   -
          activation_values                  -  array (7) - val for each observable
          observable_data       - numpy array (X,Y,7) - contains 7 observables
          DTT                   - numpy array (X,Y,7) - contains 7 DTT
          final_cloud_mask      - numpy array (X,Y)   - contains final cloudmask
          BRFs                  - numpy array (X,Y,6) - contains 6 bands of BRF
          SZA, VZA, VAA, SAA    - numpy array (X,Y)   - sun-view geometry
          scene_type_identifier - numpy array (X,Y)   - contains scene type
    RETURN:
          put inputs into an hdf5 file
    '''
    import h5py
    import numpy as np

    #save_path = ''#'../MCM_output/'
    if save_path == '':
        hf_path = save_path + 'MCM_output.HDF5'
    else:
        hf_path = save_path
    
    hf = h5py.File(hf_path, 'w')

    #create structure in hdf file
    group = hf.create_group('cloud_mask_output')
    sun_view_geometry = hf.create_group('sun_view_geometry')
    BRF = hf.create_group('Reflectance')
    Ancillary = hf.create_group('Ancillary')
    configuration_file = Ancillary.create_group('configuration_file')

    #save empty thresholds to sub group
    group.create_dataset('observable_data',\
                          data=observable_data, dtype='f', compression='gzip')
    group.create_dataset('DTT',\
                          data=DTT   , dtype='f', compression='gzip')
    group.create_dataset('final_cloud_mask',\
                          data=final_cloud_mask, dtype='f', compression='gzip')

    #some extra goodies; BRF/scene_type_identifier/SZA/VZA/SAA/VAA/congif.csv
    BRF.create_dataset('band_04', data=BRFs[:,:,0],\
                        dtype='f', compression='gzip')
    BRF.create_dataset('band_05', data=BRFs[:,:,1],\
                        dtype='f', compression='gzip')
    BRF.create_dataset('band_06', data=BRFs[:,:,2],\
                        dtype='f', compression='gzip')
    BRF.create_dataset('band_09', data=BRFs[:,:,3],\
                        dtype='f', compression='gzip')
    BRF.create_dataset('band_12', data=BRFs[:,:,4],\
                        dtype='f', compression='gzip')
    BRF.create_dataset('band_13', data=BRFs[:,:,5],\
                        dtype='f', compression='gzip')

    Ancillary.create_dataset('scene_type_identifier', data=scene_type_identifier,\
                          dtype='f', compression='gzip')

    configuration_file.create_dataset('Sun_glint_exclusion_angle',\
                                       data=Sun_glint_exclusion_angle, dtype='f')
    configuration_file.create_dataset('Max_RDQI', data=Max_RDQI,\
                                       dtype='f')
    configuration_file.create_dataset('Max_valid_DTT', data=Max_valid_DTT,\
                                       dtype='f')
    configuration_file.create_dataset('Min_valid_DTT', data=Min_valid_DTT,\
                                       dtype='f')
    configuration_file.create_dataset('fill_val_1', data=fill_val_1,\
                                       dtype='f')
    configuration_file.create_dataset('fill_val_2', data=fill_val_2,\
                                       dtype='f')
    configuration_file.create_dataset('fill_val_3', data=fill_val_3,\
                                       dtype='f')
    configuration_file.create_dataset('Min_num_of_activated_testsMin_num_of_activated_tests', data=Min_num_of_activated_tests,\
                                       dtype='f')
    configuration_file.create_dataset('activation_values', data=activation_values,\
                                       dtype='f', compression='gzip')

    sun_view_geometry.create_dataset('solar_zenith_angle', data=SZA,\
                          dtype='f', compression='gzip')
    sun_view_geometry.create_dataset('viewing_zenith_angle', data=VZA,\
                          dtype='f', compression='gzip')
    sun_view_geometry.create_dataset('solar_azimuth_angle', data=SAA,\
                          dtype='f', compression='gzip')
    sun_view_geometry.create_dataset('viewing_azimuth_angle', data=VAA,\
                          dtype='f', compression='gzip')

    #add labels for activation values
    hf['Ancillary/configuration_file/activation_values'].dims[0].label = 'WI, NDVI, NDSI, VIS Ref, NIR Ref, SVI, Cirrus'

    hf.close()

