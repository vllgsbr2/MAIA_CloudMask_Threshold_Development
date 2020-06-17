'''
Author: Javier Villegas Bravo
Date: 03/06/20

Purpose:
Confusion matrix to evaluate the performance of the MAIA Cloud Mask
'''

def scene_confusion_matrix(MOD_CM_path, MAIA_CM_path, MCM_Output_path):
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
    home        = MCM_Output_path
    time_stamps = os.listdir(MAIA_CM_path)

    #open file to write to
    with h5py.File(MCM_Output_path + 'conf_matx_scene_reproduce.HDF5', 'w') as hf_scene_level_conf_matx:

        for time_stamp in time_stamps:
        #open file one at a time according to time stamp
            with h5py.File(MOD_CM_path + '/test_JPL_data_{}.HDF5'.format(time_stamp), 'r')  as hf_MOD_CM  ,\
                 h5py.File(MAIA_CM_path + '/'+time_stamp+'/MCM_Output.HDF5', 'r') as hf_MAIA_CM:

                MAIA_CM = hf_MAIA_CM['cloud_mask_output/final_cloud_mask'][()]#output files
                MOD_CM  = hf_MOD_CM['MOD35_cloud_mask'][()]#input files

                conf_matx_mask = np.zeros((1000,1000,4), dtype=np.int)


                #both return cloudy
                true         = np.where((MAIA_CM == 0) & (MOD_CM == 0))
                #both return clear
                false        = np.where((MAIA_CM == 1) & (MOD_CM != 0))
                #MOD clear MAIA cloudy
                false_cloudy = np.where((MAIA_CM == 0) & (MOD_CM != 0))
                #MOD cloudy MAIA clear
                false_clear  = np.where((MAIA_CM == 1) & (MOD_CM == 0))

                conf_mat_table = np.array([true[0].sum(), false[0].sum(), false_cloudy[0].sum(),\
                                           false_clear[0].sum()], dtype=np.int)

                conf_matx_mask[true[0]        , true[1]        , 0] = 1
                conf_matx_mask[false[0]       , false[1]       , 1] = 1
                conf_matx_mask[false_clear[0] , false_clear[1] , 2] = 1
                conf_matx_mask[false_cloudy[0], false_cloudy[1], 3] = 1

                conf_matx_mask[MAIA_CM==3] = -999

                print(time_stamp, conf_mat_table[:2].sum()/conf_mat_table.sum())
                try:
                    hf_scene_level_conf_matx.create_dataset('confusion_matrix_table_{}'.format(time_stamp), data=conf_mat_table)
                    hf_scene_level_conf_matx.create_dataset('confusion_matrix_mask_{}'.format(time_stamp) , data=conf_matx_mask, compression='gzip')
                except:
                    hf_scene_level_conf_matx['confusion_matrix_table_{}'.format(time_stamp)][:] = conf_mat_table
                    hf_scene_level_conf_matx['confusion_matrix_mask_{}'.format(time_stamp)][:]  = conf_matx_mask

def group_confusion_matrix(hf_group, hf_thresh, hf_confmatx):
    '''
    need to grab bin to make olp from grouped data
    grab threshold for that bin; remember TA -> DOY -> (cos(SZA), VZA, RAZ, Scene_ID)
    compare obs of every data point in grouped data file bin and decide cloud/not cloud
    compare to the truth found in grouped data
    record into thresholds file
    '''

    hf_group_keys = list(hf_group.keys())
    num_groups = len(hf_group_keys)
    accuracy = np.zeros((num_groups))
    #read bin ID (OLP)
    n = 0
    OLP = np.zeros((num_groups, 4))
    for i, bins in enumerate(hf_group_keys):
        #print(bins)
        OLP[i,:] = [int(bins[n+7:n+9])  ,\
                    int(bins[n+14:n+16]),\
                    int(bins[n+21:n+23]),\
                    int(bins[n+45:n+47]) ]
    OLP = OLP.astype(dtype=np.int)

    #read in thresholds
    obs_names = ['WI', 'NDVI', 'NDSI', 'VIS_Ref', 'NIR_Ref', 'SVI', 'Cirrus']
    thresholds = np.empty((7,10,14,12,15))
    for i, obs_ in enumerate(obs_names):
        path = 'TA_bin_01/DOY_bin_06/{}'.format(obs_)
        thresholds[i] = hf_thresh[path][()]

    #iterate by group
    for i, bin_ID in enumerate(hf_group_keys):

        #number of data points in current group/bin_ID
        num_points = hf_group[bin_ID].shape[0]

        #grab obs & CM of the current group
        data = hf_group[bin_ID][()]
        cloud_mask = data[:, 0]
        obs        = data[:, 1:]

        #calculate distance to threshold for all NDxI in the current group
        DTT_NDxI = np.zeros((num_points, 2))
        for j in range(1,3):
            NDxI = obs[:,j]
            #print(j, OLP[i,0], OLP[i,1], OLP[i,2], OLP[i,3])
            T    = thresholds[j, OLP[i,0], OLP[i,1], OLP[i,2], OLP[i,3]]
            #put 0.001 instead of zero to avoid divide by zero error
            if T==0:
                T = 1e-3
            DTT_NDxI[:,j-1] = (T - np.abs(NDxI)) / T

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
                if j==0 and thresh_temp[j] >= obs[k,0] and (olp_temp[3]!=13 and olp_temp[3]!=14):
                    cloud_mask_MAIA[k] = 0
                #DTT for NDxI. Must exceed 0
                #NDVI everything but snow
                elif j==1 and olp_temp[3]!=14 and DTT_NDxI[k,0] >= 0:
                    cloud_mask_MAIA[k] = 0
                #NDSI only over snow
                elif j==2 and olp_temp[3]==14 and DTT_NDxI[k,1] >= 0:
                    cloud_mask_MAIA[k] = 0
                #VIS, NIR, Cirrus. Must exceed thresh
                #VIS applied only over land
                elif j==3 and olp_temp[3]<=11 and thresh_temp[j] <= obs[k,3]:
                    cloud_mask_MAIA[k] = 0
                #NIR only applied over water (no sunglint)
                elif j==4 and olp_temp[3]==12 and thresh_temp[j] <= obs[k,4]:
                    cloud_mask_MAIA[k] = 0
                #SVI applied over all surfaces when over 0. Must exceed thresh
                elif j==5 and obs[k,5] >= 0.0  and thresh_temp[j] <= obs[k,5]:
                    cloud_mask_MAIA[k] = 0
                #j==5 is SVI. So this is j==6 for cirrus applied everywhere
                elif j==6 and thresh_temp[j] <= obs[k,6]:
                    cloud_mask_MAIA[k] = 0
                else:
                    pass


        #compare with CM; cloudy==0, clear==1
        MOD_CM = cloud_mask
        MAIA_CM = cloud_mask_MAIA
        #both return cloudy
        true      = np.where((MAIA_CM == 0) & (MOD_CM == 0))[0].sum()
        #both return clear
        false     = np.where((MAIA_CM == 1) & (MOD_CM != 0))[0].sum()
        #MOD cloudy MAIA clear
        false_neg = np.where((MAIA_CM == 1) & (MOD_CM == 0))[0].sum()
        #MOD clear MAIA cloudy
        false_pos = np.where((MAIA_CM == 0) & (MOD_CM != 0))[0].sum()

        #make result into confusion matrix; make nan into fill val -999
        conf_mat = np.array([true, false, false_pos, false_neg])
        conf_mat[np.isnan(conf_mat)==True] = -999

        #save it back into group/threshold file
        try:
            dataset = hf_confmatx.create_dataset('confusion_matrix_{}'.format(bin_ID), data=conf_mat)
        #dataset.attrs['label'] = ['both_cloudy, both_clear, MOD_cloud_MAIA_clear, MOD_clear_MAIA_cloudy']
        except:
            hf_confmatx['confusion_matrix_{}'.format(bin_ID)][:] = conf_mat

        accuracy[i] = (conf_mat[0]+conf_mat[1])/conf_mat.sum()
        print('{},{},{},{}'.format(accuracy[i], conf_mat[0], conf_mat[1], conf_mat.sum()))

    # import matplotlib.pyplot as plt
    # plt.hist(accuracy, bins = 20)
    # plt.show()

if __name__ == '__main__':

    import h5py
    import tables
    import os
    import numpy as np
    tables.file._open_files.close_all()

    #define paths for the three databases
    home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
    grouped_path = home + 'grouped_obs_and_CM.hdf5'
    thresh_path  = home + 'thresholds_reproduce.hdf5'

    # #bin confusion matrix
    # with h5py.File(grouped_path, 'r') as hf_group,\
    #      h5py.File(thresh_path, 'r') as hf_thresh,\
    #      h5py.File(home+'conf_matx.HDF5', 'w') as hf_confmatx:
    #
    #     group_confusion_matrix(hf_group, hf_thresh, hf_confmatx)

    #scene confusion matrix
    home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
    #ok i think i will reprocess MCM_Output with the MOD35 CM
    MOD_CM_path     = home + 'JPL_data_all_timestamps'#test_JPL_data_2018053.1740.HDF5
    MAIA_CM_path    = home + 'MCM_Output'#time stamp MCM_Output.HDF5
    MCM_Output_path = home
    scene_confusion_matrix(MOD_CM_path, MAIA_CM_path, MCM_Output_path)


















        #

