import h5py
import numpy as np
import configparser
import os
import sys


config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config           = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

def check_neg_SVI_thresh():

    thresh_home  = config['supporting directories']['thresh']
    thresh_path = '{}/{}/'.format(PTA_path, thresh_home)
    thresh_files = [thresh_path + x for x in os.listdir(thresh_path)]

    fill_val = -999

    for thresh_file in thresh_files:
        with h5py.File(thresh_file, 'r') as hf_thresh:
            DOYs = list(hf_thresh['TA_bin_00'].keys())
            obs  = list(hf_thresh['TA_bin_00/' + DOYs[0]].keys())

            num_negative_SVI = 0
            num_positive_SVI = 0
            for DOY in DOYs:
                SVI_path = '{}/{}/{}'.format('TA_bin_00', DOY, obs[4])
                SVI = hf_thresh[SVI_path][()].flatten()

                # SVI = hf_thresh[SVI_path][()]
                # print(np.where((SVI<0) & (SVI != fill_val)))
                # SVI = SVI.flatten()

                neg_SVI = SVI[(SVI<0) & (SVI != fill_val)]
                num_negative_SVI += len(neg_SVI)
                num_positive_SVI += len(SVI[SVI>=0])



        print('%neg {:1.6f}, num neg {:05d}, num pos {:05d}, neg SVIs {}'.format(num_negative_SVI/num_positive_SVI, num_negative_SVI, num_positive_SVI, neg_SVI))

def check_neg_SVI_grouped():

    grouped_home  = config['supporting directories']['combined_group']
    grouped_path = '{}/{}/'.format(PTA_path, grouped_home)
    grouped_files = [grouped_path + x for x in os.listdir(grouped_path)]

    fill_val = -999

    for grouped_file in grouped_files:
        with h5py.File(grouped_file, 'r') as hf_grouped:
            bins = list(hf_grouped.keys())

            num_negative_SVI = 0
            num_positive_SVI = 0
            for bin in bins:
                # SVI is index 6
                SVI_idx = 6
                SVI     = hf_grouped[bin][:, SVI_idx].flatten()

                neg_SVI = SVI[(SVI<0) & (SVI != fill_val)]
                num_negative_SVI += len(neg_SVI)
                num_positive_SVI += len(SVI[SVI>=0])

            # print('%neg {:1.6f}, num neg {:05d}, num pos {:05d}, neg SVIs {}'.format(num_negative_SVI/num_positive_SVI, num_negative_SVI, num_positive_SVI, neg_SVI))
            print('%neg {:1.6f}, num neg {:05d}, num pos {:05d}'.format(num_negative_SVI/num_positive_SVI, num_negative_SVI, num_positive_SVI))

def check_thresh(which_thresh, flatten_or_nah=True, by_SFC_ID_or_nah=True):
    '''
    which_thresh {str} -- choose from WI,NDVI,NDSI,VIS_Ref,NIR_Ref,SVI,Cirrus
    flatten_or_nah {bool} -- True filter values and flatten; False replace w/nan
    by_SFC_ID_or_nah {bool} -- True shows thresh applied by sfcID; False all thresh
    '''

    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}

    thresh_home  = config['supporting directories']['thresh']
    thresh_path = '{}/{}/'.format(PTA_path, thresh_home)
    thresh_files = [thresh_path + x for x in os.listdir(thresh_path)]

    fill_val = -999

    for thresh_file in thresh_files:
        # print(thresh_file[-9:-3])
        with h5py.File(thresh_file, 'r') as hf_thresh:
            DOY = list(hf_thresh['TA_bin_00'].keys())[0]
            obs = list(hf_thresh['TA_bin_00/' + DOY].keys())

            thresh_path = '{}/{}/{}'.format('TA_bin_00', DOY,\
                                            obs[thresh_dict[which_thresh]])

            thresh = hf_thresh[thresh_path][()]#.flatten()

            if by_SFC_ID_or_nah:
                #axis 3 is sceneID. Store thresh when sceneID is valid for thresh
                if thresh_dict[which_thresh] == 0:
                    # WI no glint or snow
                    thresh = thresh[:,:,:,0:13]
                elif thresh_dict[which_thresh] == 1:
                    #NDVI no snow
                    thresh = thresh[:,:,:,0:14]
                elif thresh_dict[which_thresh] == 2:
                    #NDSI only snow
                    thresh = thresh[:,:,:,14]
                elif thresh_dict[which_thresh] == 3:
                    #VIS only land
                    thresh = thresh[:,:,:,0:12]
                elif thresh_dict[which_thresh] == 4:
                    #NIR only non glint water
                    thresh = thresh[:,:,:,12]
                else:
                    #everything
                    pass

            # # only take positve/non- fill_val thresholds from
            # # WI/VIS/NIR/SVI/Cirrus
            # if thresh_dict[which_thresh] >= 3 or \
            #    thresh_dict[which_thresh] == 0    :
            #
            #    if flatten_or_nah:
            #        thresh = thresh[(thresh >= 0) & (thresh != fill_val)]
            #
            # #take out fill val from NDVI/NDSI
            # else:
            #     if flatten_or_nah:
            #         thresh = thresh[thresh != fill_val]


    return thresh

def plot_thresh_hist():
    import matplotlib.pyplot as plt
    #make histograms of thresholds
    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}
    thresholds     = []
    range_ndxi     = (-1.,1.)
    range_other    = (0. ,1.5)
    num_bins_ndxi  = 100
    num_bins_other = int(num_bins_ndxi * (range_other[1] - range_other[0]) / \
                                     (range_ndxi[1]  - range_ndxi[0]))

    binned_thresholds = []
    for i, obs in enumerate(thresh_dict):
        thresholds.append(check_thresh(obs))

        print(obs)#, len(t), len(t[t<0]))

        if i==0 or i>=3:
            num_bins = num_bins_other
            range    = range_other
        else:
            num_bins = num_bins_ndxi
            range    = range_ndxi
        binned_thresholds.append(np.histogram(thresholds[i].flatten(), bins=num_bins, range=range)[0])

    f, ax = plt.subplots(ncols=4, nrows=2)

    for i, (a, obs) in enumerate(zip(ax.flat, thresh_dict)):
        if i==0 or i>=3:
            num_bins = num_bins_other
            x1, x2   = range_other
        else:
            num_bins = num_bins_ndxi
            x1, x2   = range_ndxi
        x = np.arange(x1, x2, (x2-x1)/num_bins)
        a.plot(x, binned_thresholds[i])
        a.set_title(obs)

    #only 7 obs so lets turn 8th axis off
    ax[1,3].axis('off')

    plt.show()

def plot_thresh_vs_VZA():
    import matplotlib.pyplot as plt
    #make histograms of thresholds
    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}
    fill_val = -999

    thresholds = []
    for i, obs in enumerate(thresh_dict):
        thresholds.append(check_thresh(obs, flatten_or_nah=False))

    range_ndxi     = (-1.,1.)
    range_other    = (0. ,1.5)
    num_bins_ndxi  = 100
    num_bins_other = int(num_bins_ndxi * (range_other[1] - range_other[0]) / \
                                         (range_ndxi[1]  - range_ndxi[0] ))

    f, ax = plt.subplots(ncols=4, nrows=2)

    for i, (a, obs) in enumerate(zip(ax.flat, thresh_dict)):
        # print(thresholds[i][thresholds[i]!=-999].shape)
        #make a deep copy because to not modify it
        thresh_obs_i  = np.copy(thresholds[i])
        #reorder threshold dims so VZA is first
        thresh_obs_i  = np.moveaxis(thresh_obs_i, 1, 0)
        thresh_shape  = thresh_obs_i.shape

        #reshape so VZA is axis 0 and the other axis is everything else flattened
        shape_cosSZA_x_RAZ_x_sfcID = np.prod(thresh_shape[1:])
        thresh_obs_i  = thresh_obs_i.reshape(thresh_shape[0], shape_cosSZA_x_RAZ_x_VZA)
        # thresh_obs_i  = thresh_obs_i.flatten()

        # #normalize
        # thresh_obs_i  = thresh_obs_i/thresh_obs_i.max()

        #array to match each thresh to VZA bin from [0-14]
        vza_obs_i     = np.repeat(np.arange(0,75,5), shape_cosSZA_x_RAZ_x_VZA)

        #take nan out of thresholds and adjust vza
        # vza_obs_i    = vza_obs_i[thresh_obs_i != fill_val]
        # thresh_obs_i = thresh_obs_i[thresh_obs_i != fill_val]

        #eliminate fill vals while keeping a vector for each VZA bin
        boxplot_thresh_obs_i = []
        for thresh_vza_x_i in thresh_obs_i:
            boxplot_thresh_obs_i.append(thresh_vza_x_i[thresh_vza_x_i != fill_val])

        # thresh_obs_i[thresh_obs_i == fill_val] = np.nan

        # a.scatter(vza_obs_i, thresh_obs_i)
        a.boxplot(boxplot_thresh_obs_i, notch=False, sym='')
        a.set_title(obs)

    #only 7 obs so lets turn 8th axis off
    ax[1,3].axis('off')

    plt.show()

def plot_thresh_vs_sfcID():
    import matplotlib.pyplot as plt
    #make histograms of thresholds
    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}
    fill_val = -999

    thresholds = []
    for i, obs in enumerate(thresh_dict):
        thresholds.append(check_thresh(obs, flatten_or_nah=False, by_SFC_ID_or_nah=False))

    range_ndxi     = (-1.,1.)
    range_other    = (0. ,1.5)
    num_bins_ndxi  = 100
    num_bins_other = int(num_bins_ndxi * (range_other[1] - range_other[0]) / \
                                         (range_ndxi[1]  - range_ndxi[0] ))

    f, ax = plt.subplots(ncols=4, nrows=2)

    for i, (a, obs) in enumerate(zip(ax.flat, thresh_dict)):
        #make a deep copy because to not modify it
        thresh_obs_i  = np.copy(thresholds[i])
        #reorder threshold dims so sfcID is first
        thresh_obs_i  = np.moveaxis(thresh_obs_i, -1, 0)
        thresh_shape  = thresh_obs_i.shape

        #reshape so VZA is axis 0 and the other axis is everything else flattened
        shape_cosSZA_x_RAZ_x_sfcID = int(np.prod(thresh_shape[1:]))
        thresh_obs_i  = thresh_obs_i.reshape((thresh_shape[0], shape_cosSZA_x_RAZ_x_sfcID))
        # thresh_obs_i_glint = thresh_obs_i[13,:]
        # thresh_obs_i_glint_valid_idx = np.where(thresh_obs_i_glint != fill_val)
        # print(thresh_obs_i_glint_valid_idx[0].shape)
        # print(thresh_obs_i_glint[thresh_obs_i_glint_valid_idx])

        #eliminate fill vals while keeping a vector for each VZA bin
        boxplot_thresh_obs_i = []
        for sfcID_j in range(15):
            thresh_obs_i_sfcID_j = thresh_obs_i[sfcID_j, :]
            valid_idx = np.where(thresh_obs_i_sfcID_j != fill_val)
            # if sfcID_j == 13:
            #     print(valid_idx)
            filtered_thresh_obs_i_sfcID_j = thresh_obs_i_sfcID_j[valid_idx]
            boxplot_thresh_obs_i.append(filtered_thresh_obs_i_sfcID_j)

        if i==0 or i>=3:
            ymin,ymax = range_other[0], range_other[1]
        else:
            ymin,ymax = range_ndxi[0], range_ndxi[1]

        a.set_ylim([ymin,ymax])
        a.boxplot(boxplot_thresh_obs_i, notch=False, sym='.')
        a.set_title(obs)

        #plot percent change from one sfc ID to next
        def percent_change(x, y):
            '''
            x is previous, y is next; can be arrays of same length or floats
            '''
            return 100*(y-x)/np.abs(x)

        # sfcID_thresh_percent_change = np.zeros((15))
        # for sfcID_j in range(1,15):
        #     x = np.mean(boxplot_thresh_obs_i[sfcID_j - 1])
        #     y = np.mean(boxplot_thresh_obs_i[sfcID_j])
        #
        #     p_change_temp = percent_change(x, y)
        #     # if
        #
        #     sfcID_thresh_percent_change[sfcID_j] = p_change_temp
        #     sfcID_thresh_percent_change[np.abs(sfcID_thresh_percent_change) > 100] = 100
        #
        # a_twin = a.twinx()
        # a_twin = a_twin.plot(np.arange(15), sfcID_thresh_percent_change)#, vmax=100)

        #add axis for number of samples
        # a_twin  = a.twinx()
        # num_thresh = len(boxplot_thresh_obs_i.flatten())
        # a_twin.scatter(num_thresh, )

        #color the axis' points by accuracy between 0% & 100% and add colorbar

    #only 7 obs so lets turn 8th axis off
    ax[1,3].axis('off')

    # plt.show()

def check_sunglint_thresh():
    '''
    need to check if sunglint (sfcID=13) thresholds were derived
    '''
    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}

    thresh_home  = config['supporting directories']['thresh']
    thresh_path = '{}/{}/'.format(PTA_path, thresh_home)
    thresh_files = [thresh_path + x for x in os.listdir(thresh_path)]

    fill_val = -999
    #just check in svi since it should produce something
    which_thresh = 'SVI'

    for i, thresh_file in enumerate(thresh_files):
        # print(thresh_file[-9:-3])
        with h5py.File(thresh_file, 'r') as hf_thresh:
            DOY = list(hf_thresh['TA_bin_00'].keys())[0]
            obs = list(hf_thresh['TA_bin_00/' + DOY].keys())

            thresh_path = '{}/{}/{}'.format('TA_bin_00', DOY,\
                                            obs[thresh_dict[which_thresh]])

            thresh = hf_thresh[thresh_path][()]

            sunglint_thresh       = thresh[:,:,:,13]
            valid_sunglint_thresh = sunglint_thresh[sunglint_thresh != fill_val]
            num_sunglint_thresh   = valid_sunglint_thresh.shape
            print(i, num_sunglint_thresh)

def check_sunglint_flag_in_database():
    import matplotlib.pyplot as plt

    database_home  = config['supporting directories']['database']
    database_path  = '{}/{}/'.format(PTA_path, database_home)
    database_files = [database_path + x for x in os.listdir(database_path)]

    sunglint_count = 0
    for i, db in enumerate(database_files):
        with h5py.File(db, 'r') as hf_db:
            scenes_timestamps = list(hf_db.keys())

            for scene in scenes_timestamps:
                sunglint_flag_path = '{}/{}/{}'.format(scene, 'cloud_mask', 'Sun_glint_Flag')
                sunglint_flag = hf_db[sunglint_flag_path][()]

                # sunglint_count += np.where(sunglint_flag == 0)[0].shape[0]
                print(i, scene, np.where(sunglint_flag == 0)[0].shape[0])

                if scene == '2003114.1845':
                    plt.imshow(sunglint_flag, cmap='bone_r')
                    plt.show()

def check_sunglint_flag_in_grouped_cm_and_obs():
    import matplotlib.pyplot as plt

    group_home  = config['supporting directories']['combined_group']
    group_path  = '{}/{}/'.format(PTA_path, group_home)
    group_files = [group_path + x for x in os.listdir(group_path)]

    sunglint_count = 0
    for i, group in enumerate(group_files):
        with h5py.File(group, 'r') as hf_group:
            bins = list(hf_db.keys())
            for bin_x in bins:
                if bin_x[-9:-7] == '13':
                    sunglint_bin_data = hf_group[bin]
                    print(sunglint_bin_data.shape[0])





if __name__ == '__main__':

    # check_neg_SVI_thresh()
    # check_neg_SVI_grouped()
    # plot_thresh_hist()
    # plot_thresh_vs_VZA()
    # plot_thresh_vs_sfcID()
    # check_sunglint_thresh()
    # check_sunglint_flag_in_database()
    check_sunglint_flag_in_grouped_cm_and_obs()








#
