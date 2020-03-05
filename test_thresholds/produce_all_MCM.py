from JPL_MCM_threshold_testing import MCM_wrapper
from MCM_output import make_output
import mpi4py.MPI as MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for r in range(size):
    if rank==r:
        data_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/JPL_data_all_timestamps/'
        test_data_JPL_paths = os.listdir(data_home)
        test_data_JPL_paths = [data_home + x for x in test_data_JPL_paths]

        #assign subset of files to current rank
        end               = len(test_data_JPL_paths)
        processes_per_cpu = end // (size-1)
        start             = rank * processes_per_cpu

        if rank < (size-1):
            end = (rank+1) * processes_per_cpu
        elif rank==(size-1):
            processes_per_cpu_last = end % (size-1)
            end = (rank * processes_per_cpu) + processes_per_cpu_last

        test_data_JPL_paths = test_data_JPL_paths[start:end]

        for test_data_JPL_path in test_data_JPL_paths:
            time_stamp         = test_data_JPL_path[106+14:106+26]
            Target_Area_X      = 1
            threshold_filepath = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/thresholds_MCM_efficient.hdf5'
            sfc_ID_filepath    = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/SurfaceID_LA_048.nc'
            config_filepath    = './config.csv'

            #run MCM
            Sun_glint_exclusion_angle,\
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
            scene_type_identifier = \
            MCM_wrapper(test_data_JPL_path, Target_Area_X, threshold_filepath,\
                                         sfc_ID_filepath, config_filepath)

            #save output
            #create directory for time stamp
            output_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/MCM_Output'
            directory_name = '{}/{}'.format(output_home, time_stamp)
            if not(os.path.exists(directory_name)):
                os.mkdir(directory_name)
            #save path for MCM output file
            save_path = '{}/MCM_Output.HDF5'.format(directory_name)
            make_output(Sun_glint_exclusion_angle,\
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
                        save_path=save_path)
            print(save_path)
