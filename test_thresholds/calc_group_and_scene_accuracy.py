
def scene_conf_matx_accur(conf_matx_path):
    '''
    calculate accuracy using confusion matrix files for a scene
    '''

    with h5py.File(conf_matx_path, 'r') as hf_confmatx:
        confmatx_keys = np.array(list(hf_confmatx.keys()))
        time_stamps   = [x[-12:] for x in confmatx_keys]
        masks         = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_mask']
        tables        = [x for x in confmatx_keys if x[:-13] == 'confusion_matrix_table']
        #print(masks)
        #pixel by pixel accuracy
        shape = hf_confmatx[masks[0]][()].shape
        accuracy = np.zeros(shape)
        #number of samples that contributed to every evaluation type
        #needed since not all pixels in each scene have data, -999 instead
        num_samples = np.zeros(shape)

        for i, mask in enumerate(masks):
            print(mask[-12:])
            mask = hf_confmatx[mask][()]
            present_data_idx = np.where((mask != -999) & (mask != 0))
            no_data_idx    = np.where(mask == -999)
            #set mask to zero for no data so it doesnt contribute to accuracy
            mask[no_data_idx] = 0
            #add count to num samples where data is present
            num_samples[present_data_idx] += 1

            # xx = np.shape(present_data_idx)[1]/1e6
            # print(time_stamps[i], xx)

            accuracy += mask

        #print(accuracy[:,:,0])
        #% of time mask is correct at each location, when data is present
        MCM_accuracy = np.sum(accuracy[:,:,:2], axis=2) / np.sum(num_samples, axis=2)#num_samples[:,:,0]

        return MCM_accuracy, num_samples

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
    import numpy as np
    # import mpi4py.MPI as MPI
    import configparser

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

    #file setup for group accuracy
    #where to store scene accur file
    scene_accuracy_dir  = config['supporting directories']['scene_accuracy']
    scene_accuracy_dir  = '{}/{}'.format(PTA_path, scene_accuracy_dir)
    #where to find scene confusion matricies
    conf_matx_scene_dir = config['supporting directories']['conf_matx_scene']
    conf_matx_scene_dir = '{}/{}'.format(PTA_path, conf_matx_scene_dir)

    #define file to save accur in
    scene_accuracy_save_file = '{}/{}'.format(scene_accuracy_dir, 'scene_ID_accuracy.h5')
    #list all conf matx files
    conf_matx_scene_files    = [conf_matx_scene_dir + '/' + x for x in os.listdir(conf_matx_scene_dir)]

    # DOY_bin = r
    with h5py.File(scene_accuracy_save_file, 'w') as hf_scene_accur:
        for i in range(46):
            MCM_accuracy, num_samples = scene_conf_matx_accur(conf_matx_scene_files[i])

            scene_current_group = 'DOY_bin_{:02d}'.format(i)
            hf_scene_accur.create_group(scene_current_group)
            hf_scene_accur[scene_current_group].create_dataset('MCM_accuracy', data=MCM_accuracy)
            hf_scene_accur[scene_current_group].create_dataset('num_samples' , data=num_samples)

            print('Scene DOY: {} done'.format(i))

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
