import numpy as np
import h5py

def calc_thresh(thresh_home, group_file, DOY_bin, TA, num_land_SID):
    '''
    Objective:
        Takes in grouped_obs_and_CM.hdf5 file. Inside are a datasets for
        a bin and inside are rows containing the cloud mask and
        observables for each pixel. The OLP is in the dataset name. The
        threshold is then calculated for that dataset and saved into a
        threshold file.
    Arguments:
        group_file {str} -- contains data points to calc threshold for
        all bins in the file
    Return:
        void
    '''

    DOY_end   = (DOY_bin+1)*8
    DOY_start = DOY_end - 7

    fill_val = -998

    num_samples_valid_hist = 100

    thresh_file = thresh_home + \
                  '/thresholds_DOY_{:03d}_to_{:03d}_bin_{:02d}_numSID_{:02d}.h5'\
                  .format(DOY_start, DOY_end, DOY_bin, num_land_SID)

    with h5py.File(group_file, 'r') as hf_group,\
         h5py.File(thresh_file, 'w') as hf_thresh:

        #cosSZA_00_VZA_00_RAZ_00_TA_00_sceneID_00_DOY_00
        TA_group  = hf_thresh.create_group('TA_bin_{:02d}'.format(TA))
        DOY_group = TA_group.create_group('DOY_bin_{:02d}'.format(DOY_bin))

        num_SID = num_land_SID+3
        master_thresholds = np.ones((10*15*12*num_SID)).reshape((10,15,12,num_SID))*-999
        obs_names = ['WI', 'NDVI', 'NDSI', 'VIS_Ref', 'NIR_Ref', 'SVI', 'Cirrus']
        for obs in obs_names:
            DOY_group.create_dataset(obs, data=master_thresholds)

        hf_keys    = list(hf_group.keys())
        num_points = len(hf_keys)

        for count, bin_ID in enumerate(hf_keys):
            #location in array to store threshold (cos(SZA), VZA, RAZ, Scene_ID)
            bin_idx = [int(bin_ID[7:9]), int(bin_ID[14:16]), int(bin_ID[21:23]), int(bin_ID[38:40])]

            #only calc a thresh when valid surface ID is available
            #invalid is -9
            if bin_idx[3] == -9:
                continue

            cloud_mask = hf_group[bin_ID][:,0].astype(dtype=np.int)
            obs        = hf_group[bin_ID][:,1:]

            #water can use 0 & 1 as cloudy since confidence should be good
            sfc_ID_bin = bin_idx[3]#12 is water no glint
            # if sfc_ID_bin != 12:
            clear_idx  = np.where((cloud_mask != 0) & (cloud_mask > fill_val))
            clear_obs  = obs[clear_idx[0],:]

            cloudy_idx = np.where((cloud_mask == 0) & (cloud_mask > fill_val))
            cloudy_obs = obs[cloudy_idx[0],:]
            # else:
            #     clear_idx  = np.where((cloud_mask >= 2) & (cloud_mask > fill_val))
            #     clear_obs  = obs[clear_idx[0],:]
            #
            #     cloudy_idx = np.where((cloud_mask <= 1) & (cloud_mask > fill_val))
            #     cloudy_obs = obs[cloudy_idx[0],:]

            #if there isn't enough clear or cloudy obs, assign value to make threshold true
            #if no clear, and need clear, assign threshold as least brightest cloudy
            #if no cloudy, and need cloudy, assign thresholds as 1e-3 (NDxI)

            for i in range(7):
                #path to TA/DOY/obs threshold dataset
                path = 'TA_bin_{:02d}/DOY_bin_{:02d}/{}'.format(TA, DOY_bin , obs_names[i])
                # print(path)

                #clean the obs for the thresh calculation
                clean_clear_obs = clear_obs[:,i]
                clean_clear_obs = clean_clear_obs[(clean_clear_obs > -998) & (clean_clear_obs <= 32767)]

                clean_cloudy_obs = cloudy_obs[:,i]
                clean_cloudy_obs = clean_cloudy_obs[(clean_cloudy_obs > -998) & (clean_cloudy_obs <= 32767)]

                #WI
                if i==0:
                    if clean_clear_obs.shape[0] > num_samples_valid_hist:
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] = \
                        np.nanpercentile(clean_clear_obs, 1)

                    # #choose least white cloudy pixel as threshold if no clear obs
                    # elif clean_cloudy_obs.shape[0] > num_samples_valid_hist:
                    #     hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] = \
                    #     clean_cloudy_obs.max()
                    # else:
                    #     pass

                #NDVI
                elif i==1:
                    #pick max from cloudy hist for non water
                    if clean_cloudy_obs.shape[0] > num_samples_valid_hist and bin_idx[3] != 12:
                            hist, bin_edges = np.histogram(clean_cloudy_obs, bins=128, range=(-1,1))
                            thresh_current = bin_edges[1:][hist==hist.max()].min()

                            hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] =\
                            thresh_current
                    #pick 99% clear for water since water gives negative values
                    elif clean_clear_obs.shape[0] > num_samples_valid_hist and bin_idx[3] == 12:
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] =\
                        np.nanpercentile(clean_clear_obs, 99)
                    else:
                        pass

                    # #set default value of 1e-3 if no cloudy obs available
                    # else:
                    #     hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] = 1e-3

                #NDSI
                elif i==2:
                    #pick 1% clear for snow since snow gives > 0.4 NDSI
                    if clean_clear_obs.shape[0] > num_samples_valid_hist:
                        if bin_idx[3] == 14:
                            hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]] =\
                            np.nanpercentile(clean_clear_obs, 1)
                        else:
                            pass
                #VIS/NIR/SVI/Cirrus
                else:
                    if clean_clear_obs.shape[0] > num_samples_valid_hist:
                        # clean_clear_obs = clean_clear_obs[(clean_clear_obs>=0) & (clean_clear_obs < 1)]
                        current_thresh = np.nanpercentile(clean_clear_obs, 99)
                        hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2],\
                                        bin_idx[3]] = current_thresh

                    # else:
                    #     if clean_cloudy_obs.shape[0] > num_samples_valid_hist:
                    #         current_thresh = clean_cloudy_obs.min()
                    #         hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2],\
                    #                         bin_idx[3]] = current_thresh

                # current_thresh = hf_thresh[path][bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3]]
                # if np.abs(current_thresh) > 2:
                #     debug_line = 'cos(SZA): {:02d} VZA: {:02d} RAA: {:02d} SID: {:02d} thresh: {:3.3f}'.\
                #               format(bin_idx[0], bin_idx[1], bin_idx[2], bin_idx[3], current_thresh)
                #     print(debug_line)
if __name__ == '__main__':

    import h5py
    import tables
    import os
    import mpi4py.MPI as MPI
    import sys
    import configparser
    tables.file._open_files.close_all()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:

            config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
            config = configparser.ConfigParser()
            config.read(config_home_path+'/test_config.txt')

            PTA          = config['current PTA']['PTA']
            PTA_path     = config['PTAs'][PTA]
            TA           = int(config['Target Area Integer'][PTA])
            grouped_home = config['supporting directories']['combined_group']
            thresh_home  = config['supporting directories']['thresh']
            grouped_home = '{}/{}'.format(PTA_path, grouped_home)
            thresh_home  = '{}/{}'.format(PTA_path, thresh_home)

            #define paths for the database
            DOY_bin   = r
            DOY_end   = (DOY_bin+1)*8
            DOY_start = DOY_end - 7
            grouped_file_path = '{}/grouped_obs_and_CM_{:03d}_to_{:03d}_bin_{:02d}.h5'.\
                                format(grouped_home, DOY_start, DOY_end, DOY_bin)

            num_land_SID = int(sys.argv[1])

            calc_thresh(thresh_home, grouped_file_path, DOY_bin, TA, num_land_SID)
