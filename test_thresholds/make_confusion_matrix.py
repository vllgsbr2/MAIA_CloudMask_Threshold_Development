'''
Author: Javier Villegas Bravo
Date: 03/06/20

Purpose:
Confusion matrix to evaluate the performance of the MAIA Cloud Mask
'''

def scene_confusion_matrix(MOD_CM_path, MAIA_CM_path, DOY_bin, conf_matx_scene_path):
    '''
    #now for a particular scene calculate the confusion matrix
    #Grab final cloud mask
    #grab modis cloud mask
    #compare cloudy to condient cloudy, and clear to all modis clear designations
    #record into MCM output
    '''

    #conf_matx_mask = np.zeros((1000,1000,4), dtype=np.int)
    #conf_matx_mask[MAIA_CM==-999] = -999

    #just some hous keeping to avoid redundent definitions
    time_stamps = os.listdir(MAIA_CM_path)
    #select time stamps in current DOY bin
    DOY_end     = (DOY_bin + 1)*8
    DOY_start   = DOY_end - 7
    time_stamps = [t for t in time_stamps if int(t[4:7]) >= DOY_start and int(t[4:7]) <= DOY_end]
    #use years 2004/2018/2010 which are the test set
    time_stamps = [x for x in time_stamps if int(x[:4])==2004 or int(x[:4])==2010 or int(x[:4])==2018]

    with h5py.File('{}/conf_matx_scene_DOY_bin_{:02d}.h5'.format(conf_matx_scene_path, DOY_bin), 'w') as hf_scene_level_conf_matx:

        for time_stamp in time_stamps:
        #open file one at a time according to time stamp
            with h5py.File('{}/test_JPL_data_{}.h5'.format(MOD_CM_path, time_stamp), 'r')  as hf_MOD_CM  ,\
                 h5py.File('{}/{}/MCM_Output.h5'.format(MAIA_CM_path, time_stamp), 'r') as hf_MAIA_CM:

                MAIA_CM = hf_MAIA_CM['cloud_mask_output/final_cloud_mask'][()]#output files
                MOD_CM  = hf_MOD_CM['MOD35_cloud_mask'][()]#input files

                shape = MAIA_CM.shape
                conf_matx_mask = np.zeros((shape[0],shape[1],4), dtype=np.int)

                #both return cloudy
                true         = np.where((MAIA_CM == 0) & (MOD_CM == 0))
                #both return clear
                false        = np.where((MAIA_CM == 1) & (MOD_CM != 0))
                #MOD clear MAIA cloudy
                false_cloudy = np.where((MAIA_CM == 0) & (MOD_CM != 0))
                #MOD cloudy MAIA clear
                false_clear  = np.where((MAIA_CM == 1) & (MOD_CM == 0))

                conf_matx_mask[true[0]        , true[1]        , 0] = 1
                conf_matx_mask[false[0]       , false[1]       , 1] = 1
                conf_matx_mask[false_clear[0] , false_clear[1] , 2] = 1
                conf_matx_mask[false_cloudy[0], false_cloudy[1], 3] = 1

                conf_mat_table = np.array([conf_matx_mask[true].sum(), conf_matx_mask[false].sum(), conf_matx_mask[false_cloudy].sum(),\
                                           conf_matx_mask[false_clear].sum()], dtype=np.int)

                print(time_stamp, conf_mat_table[:2].sum()/conf_mat_table.sum())

                conf_matx_mask[MAIA_CM >= 2] = -999

                try:
                    hf_scene_level_conf_matx.create_dataset('confusion_matrix_table_{}'.format(time_stamp), data=conf_mat_table)
                    hf_scene_level_conf_matx.create_dataset('confusion_matrix_mask_{}'.format(time_stamp) , data=conf_matx_mask, compression='gzip')
                except:
                    hf_scene_level_conf_matx['confusion_matrix_table_{}'.format(time_stamp)][:] = conf_mat_table
                    hf_scene_level_conf_matx['confusion_matrix_mask_{}'.format(time_stamp)][:]  = conf_matx_mask

def group_confusion_matrix(hf_group, hf_thresh, hf_confmatx, num_land_sfc_types, DOY_bin, Target_Area_X):
    '''
    need to grab bin to make olp from grouped data
    grab threshold for that bin; remember TA -> DOY -> (cos(SZA), VZA, RAZ, Scene_ID)
    compare obs of every data point in grouped data file bin and decide cloud/not cloud
    compare to the truth found in grouped data
    record into thresholds file
    '''

    hf_group_keys = list(hf_group.keys())
    num_groups    = len(hf_group_keys)
    accuracy      = np.zeros((num_groups))

    #read bin ID (OLP)
    n   = 0
    OLP = np.zeros((num_groups, 4))
    for i, bins in enumerate(hf_group_keys):
        print(bins)
        cosSZA   = int(bins[7:9])
        VZA      = int(bins[14:16])
        RAZ      = int(bins[21:23])
        Scene_ID = int(bins[38:40])

        OLP[i,:] = [cosSZA , VZA, RAZ, Scene_ID]

    OLP = OLP.astype(dtype=np.int)

    #read in thresholds
    obs_names = ['WI', 'NDVI', 'NDSI', 'VIS_Ref', 'NIR_Ref', 'SVI', 'Cirrus']
    thresholds = np.empty((7,10,15,12,15))
    for i, obs_ in enumerate(obs_names):
        path = 'TA_bin_{:02d}/DOY_bin_{:02d}/{}'.format(Target_Area_X, DOY_bin, obs_)
        thresholds[i] = hf_thresh[path][()]

    #define surface types by bin number
    #remember it's 0 indexed
    water     = num_land_sfc_types + 0 #12
    sun_glint = num_land_sfc_types + 1 #13
    snow      = num_land_sfc_types + 2 #14

    #iterate by group
    for i, bin_ID in enumerate(hf_group_keys):
        #skip scene ID -9 since it has no threshold or validity as a group
        if Scene_ID != -9:

            #number of data points in current group/bin_ID
            num_points = hf_group[bin_ID].shape[0]

            #grab obs & CM of the current group
            data = hf_group[bin_ID][()]
            cloud_mask = data[:, 0]
            obs        = data[:, 1:]

            #calculate distance to threshold NDVI in the current group
            #NDVI is 1; only perform over non water
            if OLP[i,3] != 12:
                NDVI = obs[:,1]
                T    = thresholds[1, OLP[i,0], OLP[i,1], OLP[i,2], OLP[i,3]]
                #put 0.001 instead of zero to avoid divide by zero error
                if T==0:
                    T = 1e-3
                DTT_NDVI = (T - np.abs(NDVI)) / T

            #see if any obs trigger cloudy
            #assume clear (i.e. 1), cloudy is 0
            cloud_mask_MAIA = np.ones((num_points))

            #[WI_0, NDVI_1, NDSI_2, VIS_3, NIR_4, SVI_5, Cirrus_6]
            #water = 12 / sunglint over water = 13/ snow =14 / land = 0-11
            olp_temp = OLP[i,:]
            thresh_temp = thresholds[:, olp_temp[0], olp_temp[1], olp_temp[2], olp_temp[3]]
            for j in range(7):
                for k in range(num_points):
                    #this is for whiteness since 0 is whiter than 1 and not applied over snow/ice or sunglint
                    if j==0 and thresh_temp[j] >= obs[k,j] and (olp_temp[3]!=snow and olp_temp[3]!=sun_glint):
                        cloud_mask_MAIA[k] = 0
                    #DTT for NDxI. Must exceed 0
                    #NDVI everything but snow
                    elif j==1 and olp_temp[3] != snow:
                        #over non water use original DTT
                        if olp_temp[3] != 12 and DTT_NDVI[k] >= 0:
                            cloud_mask_MAIA[k] = 0
                        #over water obs just needs to exceed the thresh
                        elif thresh_temp[j] <= obs[k,j] and olp_temp[3] == 12:
                            cloud_mask_MAIA[k] = 0

                    #NDSI only over snow; just like whiteness test
                    elif j==2 and olp_temp[3] == snow and thresh_temp[j] >= obs[k,j]:
                        cloud_mask_MAIA[k] = 0
                    #VIS, NIR, Cirrus. Must exceed thresh
                    #VIS applied only over land
                    elif j==3 and olp_temp[3] < water and thresh_temp[j] <= obs[k,j]:
                        cloud_mask_MAIA[k] = 0
                    #NIR only applied over water (no sunglint)
                    elif j==4 and olp_temp[3] == water and thresh_temp[j] <= obs[k,j]:
                        cloud_mask_MAIA[k] = 0
                    #SVI applied over all surfaces when over 0. Must exceed thresh
                    elif j==5 and obs[k,5] >= 0.0 and thresh_temp[j] <= obs[k,j]:
                        cloud_mask_MAIA[k] = 0
                    #j==6 for cirrus applied everywhere
                    elif j==6 and thresh_temp[j] <= obs[k,6]:
                        cloud_mask_MAIA[k] = 0
                    else:
                        pass


            #compare with CM; cloudy==0, clear==1
            MOD_CM  = cloud_mask
            MAIA_CM = cloud_mask_MAIA
            #both return cloudy
            true      = np.where((MAIA_CM == 0) & (MOD_CM == 0))[0].sum()
            #both return clear
            false     = np.where((MAIA_CM == 1) & (MOD_CM != 0))[0].sum()
            #MOD clear MAIA cloudy
            false_pos = np.where((MAIA_CM == 0) & (MOD_CM != 0))[0].sum()
            #MOD cloudy MAIA clear
            false_neg = np.where((MAIA_CM == 1) & (MOD_CM == 0))[0].sum()


            #make result into confusion matrix; make nan into fill val -999
            conf_mat = np.array([true, false, false_pos, false_neg])
            conf_mat[np.isnan(conf_mat)==True] = -999

            #save it back into group/threshold file
            try:
                dataset = hf_confmatx.create_dataset('confusion_matrix_{}'.format(bin_ID), data=conf_mat)
            #dataset.attrs['label'] = ['both_cloudy, both_clear, MOD_cloud_MAIA_clear, MOD_clear_MAIA_cloudy']
            except:
                hf_confmatx['confusion_matrix_{}'.format(bin_ID)][:] = conf_mat

            conf_mat_sum = conf_mat.sum()
            if conf_mat_sum != 0 and np.isnan(conf_mat_sum) == False:
                accuracy[i] = (conf_mat[0]+conf_mat[1])/conf_mat_sum
            else:
                accuracy[i] = np.nan

            print('{},{},{},{}'.format(accuracy[i], conf_mat[0], conf_mat[1], conf_mat_sum))

    # import matplotlib.pyplot as plt
    # plt.hist(accuracy, bins = 20)
    # plt.show()

if __name__ == '__main__':

    import h5py
    import tables
    import os
    import numpy as np
    import configparser
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
            config           = configparser.ConfigParser()
            config.read(config_home_path+'/test_config.txt')

            PTA           = config['current PTA']['PTA']
            PTA_path      = config['PTAs'][PTA]
            Target_Area_X = int(config['Target Area Integer'][PTA])

            calc_scene  = True
            group_accur = True

            DOY_bin = rank

            if calc_scene:
                #scene confusion matrix ****************************************
                #define paths for the three databases
                MOD_CM_path          = PTA_path + '/' + config['supporting directories']['MCM_Input']
                MAIA_CM_path         = PTA_path + '/' + config['supporting directories']['MCM_Output']
                conf_matx_scene_path = PTA_path + '/' + config['supporting directories']['conf_matx_scene']

                scene_confusion_matrix(MOD_CM_path, MAIA_CM_path, DOY_bin, conf_matx_scene_path)

            if group_accur:
                #bin confusion matrix ******************************************
                grouped_path   = PTA_path + '/' + config['supporting directories']['combined_group']
                thresh_path    = PTA_path + '/' + config['supporting directories']['thresh']
                conf_matx_path = PTA_path + '/' + config['supporting directories']['conf_matx_group']

                grouped_files   = [grouped_path   + '/' + x for x in np.sort(os.listdir(grouped_path))]
                thresh_files    = [thresh_path    + '/' + x for x in np.sort(os.listdir(thresh_path))]

                conf_matx_filepath  = '{}/conf_matx_group_DOY_bin_{:02d}.h5'.format(conf_matx_path, DOY_bin)

                num_land_sfc_types = 12

                with h5py.File(grouped_files[DOY_bin] , 'r') as hf_group,\
                     h5py.File(thresh_files[DOY_bin]  , 'r') as hf_thresh,\
                     h5py.File(conf_matx_filepath     , 'w') as hf_confmatx:

                    group_confusion_matrix(hf_group, hf_thresh, hf_confmatx, num_land_sfc_types, DOY_bin, Target_Area_X)


















        #
