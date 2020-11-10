
def scene_conf_matx_accur(conf_matx_path, SID, numKmeansSID):
    '''
    calculate accuracy using confusion matrix files for a scene
    '''

    with h5py.File(conf_matx_path, 'r') as hf_confmatx:
        confmatx_keys = np.array(list(hf_confmatx.keys()))
        time_stamps   = [x[-12:] for x in confmatx_keys]
        masks         = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_mask']
        tables        = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_table']

        #pixel by pixel accuracy
        shape = hf_confmatx[masks[0]][()].shape
        accuracy = np.zeros(shape)
        #number of samples that contributed to every evaluation type
        #needed since not all pixels in each scene have data, -999 instead
        num_samples = np.zeros(shape)

        for i, mask in enumerate(masks):

            mask = hf_confmatx[mask][()]
            #mark missing data
            no_data_idx      = np.where(mask == -999)
            #eliminate SID not from Kmeans alg.
            no_KmeansSID_idx = np.where((SID >= numKmeansSID) & (SID == -9))
            #set mask to zero so it doesnt contribute to accuracy
            mask[no_data_idx]      = 0
            mask[no_KmeansSID_idx] = 0
            #add count to num samples where data is present
            present_data_idx = np.where(mask != 0)
            num_samples[present_data_idx] += 1

            accuracy += mask

        true_cloud  = np.sum(accuracy[:,:,0])
        true_clear  = np.sum(accuracy[:,:,1])
        false_clear = np.sum(accuracy[:,:,2])
        false_cloud = np.sum(accuracy[:,:,3])

        total_conf_matx = [true_cloud, true_clear, false_clear, false_cloud]
        #% of time mask is correct at each location, when data is present
        total_sum = np.sum(num_samples, axis=2)
        MCM_accuracy = np.sum(accuracy[:,:,:2], axis=2) / total_sum

        #take care of inf values when dividing by zero if applicable
        MCM_accuracy[total_sum == 0] = -999

        return MCM_accuracy, num_samples, total_conf_matx

def group_conf_matx_accur(conf_matx_path):
    '''
    calculate accuracy using confusion matrix files for a group
    '''
    accuracy_of_groups = {}
    with h5py.File(conf_matx_path, 'r') as hf_conf_matx_group:
        groups = list(hf_conf_matx_group.keys())

        for group in groups:
            #conf matx contains a 4 number array with counts of
            #[true, false, false_pos, false_neg]
            conf_matx_temp = hf_conf_matx_group[group][()]
            num_correct    = np.sum(conf_matx_temp[:2])
            num_total      = np.sum(conf_matx_temp)
            #store accuracy and num of total samples
            accuracy_of_groups[group] = np.array([num_correct/num_total, num_total])

    return accuracy_of_groups

if __name__ == '__main__':

    import h5py
    import os
    import sys
    import numpy as np
    # import mpi4py.MPI as MPI
    import configparser
    from netCDF4 import Dataset

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # for r in range(size):
    #     if rank==r:

    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config           = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA           = config['current PTA']['PTA']
    PTA_path      = config['PTAs'][PTA]
    # Target_Area_X = int(config['Target Area Integer'][PTA])

    numKmeansSID = int(sys.argv[1])

    #file setup for scene accuracy
    #where to store scene accur file
    scene_accuracy_dir  = config['supporting directories']['scene_accuracy']
    scene_accuracy_dir  = '{}/{}'.format(PTA_path, scene_accuracy_dir)
    #where to find scene confusion matricies
    conf_matx_scene_dir = config['supporting directories']['conf_matx_scene']
    conf_matx_scene_dir = '{}/{}/numKmeansSID_{:02d}'.format(PTA_path, conf_matx_scene_dir, numKmeansSID)

    #define file to save accur in
    scene_accuracy_save_dir = '{}/numKmeansSID_{:02d}'.format(scene_accuracy_dir, numKmeansSID)
    scene_accuracy_save_file = '{}/numKmeansSID_{:02d}/{}'.format(scene_accuracy_dir, numKmeansSID,'scene_ID_accuracy.h5')
    if not(os.path.exists(scene_accuracy_save_dir)):
        os.mkdir(scene_accuracy_save_dir)
    #list all conf matx files
    conf_matx_scene_files    = [conf_matx_scene_dir + '/' + x for x in os.listdir(conf_matx_scene_dir)]

    #define SID file base path
    sfc_ID_home = '{}/{}'.format(PTA_path, config['supporting directories']['Surface_IDs'])

    # DOY_bin = r
    total_conf_matx = np.array([0.,0.,0.,0.])
    with h5py.File(scene_accuracy_save_file, 'w') as hf_scene_accur:
        for i in range(46):
            DOY_end = (i+1)*8
            SID_file    = 'num_Kmeans_SID_{:02d}/surfaceID_LosAngeles_{:03d}.nc'.format(numKmeansSID, DOY_end)
            sfc_ID_filepath    = '{}/{}'.format(sfc_ID_home, SID_file)
            with Dataset(sfc_ID_filepath, 'r') as nc_SID:
                SID = nc_SID.variables['surface_ID'][:,:]
            MCM_accuracy, num_samples, conf_matx_x = scene_conf_matx_accur(conf_matx_scene_files[i], SID, numKmeansSID)
            total_conf_matx += conf_matx_x
            print(conf_matx_x)
            scene_current_group = 'DOY_bin_{:02d}'.format(i)
            hf_scene_accur.create_group(scene_current_group)
            hf_scene_accur[scene_current_group].create_dataset('MCM_accuracy', data=MCM_accuracy)
            hf_scene_accur[scene_current_group].create_dataset('num_samples' , data=num_samples)
            hf_scene_accur[scene_current_group].create_dataset('total_conf_matx' , data=conf_matx_x)

            print('Scene DOY: {} done'.format(i))

    print(total_conf_matx)

    sys.exit()

    #****************************group******************************************

    #file setup for group accuracy
    #where to store group accur file
    group_accuracy_dir  = config['supporting directories']['group_accuracy']
    group_accuracy_dir  = '{}/{}'.format(PTA_path, group_accuracy_dir)
    #where to find group confusion matricies
    conf_matx_group_dir = config['supporting directories']['conf_matx_group']
    conf_matx_group_dir = '{}/{}'.format(PTA_path, conf_matx_group_dir)

    #define file to save accur in
    group_accuracy_save_file = '{}/{}'.format(group_accuracy_dir, 'group_ID_accuracy.h5')
    # list all conf matx files
    conf_matx_group_files    = [conf_matx_group_dir + '/' + x for x in os.listdir(conf_matx_group_dir)]

    with h5py.File(group_accuracy_save_file, 'w') as hf_group_accur:
        for i in range(46):
            accuracy_of_groups = group_conf_matx_accur(conf_matx_group_files[i])

            for group, accur_num_samples in accuracy_of_groups.items():
                hf_group_accur.create_group(group)
                hf_group_accur[group].create_dataset('accuracy', data=accur_num_samples[0])
                hf_group_accur[group].create_dataset('num_samples', data=accur_num_samples[1], dtype='int')

            print('Group DOY: {} done'.format(i))

















            #
