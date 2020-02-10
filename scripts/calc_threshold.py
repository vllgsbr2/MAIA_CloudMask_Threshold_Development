import numpy as np
import h5py

def calc_thresh(group_file):
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
    home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
    with h5py.File(group_file, 'r+') as hf_group,\
         h5py.File(home + '/thresholds.hdf5', 'w') as hf_thresh:

        #cosSZA_00_VZA_00_RAZ_00_TA_00_sceneID_00_DOY_00
        TA_group  = hf_thresh.create_group(bin_ID[24:29])
        DOY_group = TA_group.create_group(bin_ID[-6:])

        hf_keys    = list(hf_group.keys())
        num_points = len(hf_keys)

        for count, bin_ID in enumerate(hf_keys):
            #observables = ['WI', 'NDVI', 'NDSI', 'visRef', 'nirRef', 'SVI', 'cirrus']
            thresholds = np.zeros((7))

            cloud_mask = hf_group[bin_ID][:,0].astype(dtype=np.int)
            obs        = hf_group[bin_ID][:,1:]


            clear_idx = np.where(cloud_mask != 0)
            clear_obs = obs[clear_idx[0],:]

            cloudy_idx = np.where(cloud_mask != 0)
            cloudy_obs = obs[cloudy_idx[0],1:3] #[1:3] since we only need for NDxI

            for i in range(7):
                #WI
                if i==0:
                    thresholds[i] = np.nanpercentile(clear_obs[:,i], 1)
                #NDxI
                #pick max from cloudy hist
                elif i==1 or i==2:
                    hist, bin_edges = np.histogram(cloudy_obs[:,i-1], bins=128, range=(-1,1))
                    thresholds[i]   = bin_edges[1:][hist==hist.max()].min()
                #VIS/NIR/SVI/Cirrus
                else:
                    thresholds[i] = np.nanpercentile(clear_obs[:,i], 99)

            dataset_name = bin_ID[:23]+bin_ID[-18:-7]

            #could be an alternative to the try and except statement below
            #h5py.require_dataset(dataset_name)

            try:
                dataset = DOY_group.create_dataset(bin_ID[:23]+bin_ID[-18:-7], data=thresholds)

                #label the data
                dataset.attrs['threshold labels'] = 'WI, NDVI, NDSI, visRef,\
                                                     nir_Ref, SVI, cirrus'
            except:
                path = '{}/{}/{}'.format(bin_ID[24:29], bin_ID[-6:], bin_ID[:23]+bin_ID[-18:-7])
                hf_thresh[path][:] = thresholds

            #print(count, thresholds)

if __name__ == '__main__':

    import h5py
    import tables
    import os
    tables.file._open_files.close_all()

    #define paths for the database
    home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/'
    grouped_file_path    = home + 'grouped_obs_and_CM.hdf5'

    calc_thresh(grouped_file_path)
