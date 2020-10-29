from JPL_MCM_threshold_testing import MCM_wrapper
from MCM_output import make_output
import mpi4py.MPI as MPI
import os
import numpy as np
import configparser
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for r in range(size):
    if rank==r:
        print(1)

        config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
        config = configparser.ConfigParser()
        config.read(config_home_path+'/test_config.txt')

        PTA          = config['current PTA']['PTA']
        PTA_path     = config['PTAs'][PTA]

        num_land_SID = int(sys.argv[1])

        data_home = '{}/{}/'.format(PTA_path, config['supporting directories']['MCM_Input'])
        test_data_JPL_paths = os.listdir(data_home)
        time_stamps         = [x[14:26] for x in test_data_JPL_paths]
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
        time_stamps = time_stamps[start:end]

        for test_data_JPL_path, time_stamp in zip(test_data_JPL_paths, time_stamps):

            output_home = '{}/{}'.format(PTA_path, config['supporting directories']['MCM_Output'])
            directory_name = '{}/{}'.format(output_home, time_stamp)
            if not(os.path.exists(directory_name)):

                # print(time_stamp,test_data_JPL_path[-30:] )
                Target_Area_X      = int(config['Target Area Integer'][PTA])
                config_filepath    = './config.csv'
                DOY       = int(time_stamp[4:7])

                DOY_bins  = np.arange(8,376,8)
                DOY_bin   = np.digitize(DOY, DOY_bins, right=True)
                DOY_end   = (DOY_bin+1)*8
                DOY_start = DOY_end - 7

                print('DOY {} DOY_start {} DOY_end {} DOY_bin {}'.format(DOY, DOY_start, DOY_end, DOY_bin))
                thresh_home = '{}/{}'.format(PTA_path, config['supporting directories']['thresh'])
                threshold_filepath = '{}/thresholds_DOY_{:03d}_to_{:03d}_bin_{:02d}_numSID_{:02d}.h5'\
                                     .format(thresh_home, DOY_start, DOY_end, DOY_bin, num_land_SID)

                # sfc_ID_home = '{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'])
                # sfc_ID_filepath    = '{}/surfaceID_{}_{:03d}.nc'.format(sfc_ID_home, PTA, DOY_end)

                #for testing with many SIDs
                sfc_ID_path = '/data/gdi/c/gzhao1/MCM-surfaceID/SfcID/LosAngeles'
                sfc_ID_path  = '{}/{}/'.format(sfc_ID_path, num_land_SID)
                sfc_ID_filepath    = '{}/surfaceID_{}_{:03d}.nc'.format(sfc_ID_path, PTA, DOY_end)

                print(Target_Area_X, threshold_filepath, sfc_ID_filepath)

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
                scene_type_identifier,\
                T = \
                MCM_wrapper(test_data_JPL_path, Target_Area_X, threshold_filepath,\
                                             sfc_ID_filepath, config_filepath, num_land_sfc_types)

                #save output
                #create directory for time stamp
                output_home = '{}/{}'.format(PTA_path, config['supporting directories']['MCM_Output'])
                directory_name = '{}/{}'.format(output_home, time_stamp)
                if not(os.path.exists(directory_name)):
                    os.mkdir(directory_name)
                #save path for MCM output file
                save_path = '{}/MCM_Output.h5'.format(directory_name)
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
                # print(save_path)
