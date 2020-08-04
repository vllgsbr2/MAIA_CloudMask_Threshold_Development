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


# thresh_path = '{}/{}/{}'.format(PTA_path, thresh_home, 'thresholds_DOY_041_to_048_bin_05.h5')

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

def check_thresh(which_thresh, flatten_or_nah=True):
    '''
    which_thresh {str} -- choose from WI,NDVI,NDSI,VIS_Ref,NIR_Ref,SVI,Cirrus
    flatten_or_nah {bool} -- True filter values and flatten; False replace w/nan
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
            # only take positve/non- fill_val thresholds from
            # WI/VIS/NIR/SVI/Cirrus
            if thresh_dict[which_thresh] >= 3 or \
               thresh_dict[which_thresh] == 0    :

               if flatten_or_nah:
                   thresh = thresh[(thresh >= 0) & (thresh != fill_val)]
               else:
                   thresh[(thresh >= 0) & (thresh != fill_val)] = fill_val#np.nan

            #take out fill val from NDVI/NDSI
            else:
                if flatten_or_nah:
                    thresh = thresh[thresh != fill_val]
                else:
                    thresh[thresh != fill_val] = fill_val

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
        print(type(thresholds[i]))
        #make a deep copy because to not modify it
        thresh_obs_i  = np.copy(thresholds[i])
        #reorder threshold dims so VZA is first
        thresh_obs_i  = np.moveaxis(thresh_obs_i, 1, 0)
        thresh_shape  = thresh_obs_i.shape

        #reshape so VZA is axis 0 and the other axis is everything else flattened
        shape_cosSZA_x_RAZ_x_sfcID = np.prod(thresh_shape[1:])
        thresh_obs_i  = thresh_obs_i.reshape(thresh_shape[0], shape_cosSZA_x_RAZ_x_sfcID)
        thresh_obs_i  = thresh_obs_i.flatten()

        # #normalize
        # thresh_obs_i  = thresh_obs_i/thresh_obs_i.max()

        #array to match each thresh to VZA bin from [0-14]
        vza_obs_i     = np.repeat(np.arange(0,75,5), shape_cosSZA_x_RAZ_x_sfcID)

        #take nan out of thresholds and adjust vza
        vza_obs_i    = vza_obs_i[thresh_obs_i != fill_val]
        thresh_obs_i = thresh_obs_i[thresh_obs_i != fill_val]

        print('VZA\n', vza_obs_i)
        print('thresh\n', thresh_obs_i)
        import sys
        sys.exit()

        a.scatter(vza_obs_i, thresh_obs_i)
        a.set_title(obs)

    plt.show()

if __name__ == '__main__':

    # check_neg_SVI_thresh()
    # check_neg_SVI_grouped()
    # plot_thresh_hist()
    plot_thresh_vs_VZA()










#
