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
    thresh_files = [thresh_path + x for x in os.listdir(thresh_path) if x[:15]=='thresholds_DOY_']

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
    thresh_files = [thresh_path + x for x in os.listdir(thresh_path) if x[:15]=='thresholds_DOY_']

    fill_val = -999

    thresh = []
    for thresh_file in thresh_files:
        with h5py.File(thresh_file, 'r') as hf_thresh:
            DOY = list(hf_thresh['TA_bin_00'].keys())[0]
            obs = list(hf_thresh['TA_bin_00/' + DOY].keys())

            thresh_path = '{}/{}/{}'.format('TA_bin_00', DOY,\
                                            obs[thresh_dict[which_thresh]])
            thresh.append(hf_thresh[thresh_path][()])

    thresh = np.array(thresh)

    return thresh #thresh (DOY, cos(SZA), VZA, RAZ, SID)
    # thresh_no_DOY_dim = thresh[0]
    # for DOY in range(1, thresh.shape[0]):
    #     thresh_no_DOY_dim = np.stack((thresh_no_DOY_dim, thresh[DOY]), axis=0)

    #return thresh_no_DOY_dim

            # if by_SFC_ID_or_nah:
            #     #axis 3 is sceneID. Store thresh when sceneID is valid for thresh
            #     if thresh_dict[which_thresh] == 0:
            #         # WI no glint or snow
            #         thresh = thresh[:,:,:,0:13]
            #     elif thresh_dict[which_thresh] == 1:
            #         #NDVI no snow
            #         thresh = thresh[:,:,:,0:14]
            #     elif thresh_dict[which_thresh] == 2:
            #         #NDSI only snow
            #         thresh = thresh[:,:,:,14]
            #     elif thresh_dict[which_thresh] == 3:
            #         #VIS only land
            #         thresh = thresh[:,:,:,0:12]
            #     elif thresh_dict[which_thresh] == 4:
            #         #NIR only non glint water
            #         thresh = thresh[:,:,:,12]
            #     else:
            #         #everything
            #         pass

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


    # return thresh

def plot_thresh_hist():
    import matplotlib.pyplot as plt
    #make histograms of thresholds
    thresh_dict = {'WI':0, 'NDVI':1, 'NDSI':2, 'VIS_Ref':3, 'NIR_Ref':4,\
                   'SVI':5, 'Cirrus':6}



    #plot

    #get color cycle tool to plot rainbow ordered lines
    from matplotlib.pyplot import cm
    num_SID = 15
    color    = cm.rainbow(np.linspace(0,1,num_SID))

    master_thresh = []
    for i, obs in enumerate(thresh_dict):
        master_thresh.append(check_thresh(obs))
    master_thresh = np.array(master_thresh)
    cosSZA_bin = 5

    for DOY_bin in range(46):
        f, ax = plt.subplots(ncols=4, nrows=2, figsize=(25,13))
        for k in range(num_SID):
            #collect thresholds for each obs for just 1 SID and bin them
            binned_thresholds = []
            thresholds        = []
            for i, obs in enumerate(thresh_dict):
                #choose kth surface type
                temp_thresh = master_thresh[i, DOY_bin,:cosSZA_bin,:,:,k]

                temp_thresh = temp_thresh[(temp_thresh > -998) & (temp_thresh < 32767)]
                thresholds.append(temp_thresh)

                range_ndxi     = (-1, 1)
                range_other    = (0., 1.4)

                num_bins_ndxi  = 70
                num_bins_other = int(num_bins_ndxi * \
                (range_other[1] - range_other[0]) / (range_ndxi[1]  - range_ndxi[0]))

                if i==0 or i>=3:
                    num_bins = num_bins_other
                    range_    = range_other
                else:
                    num_bins = num_bins_ndxi
                    range_    = range_ndxi
                binned_thresholds.append(np.histogram(thresholds[i].flatten(), bins=num_bins, range=range_, density=True)[0])

            temp_thresh = binned_thresholds
            land    = list(np.arange(11))
            water   = [12]
            glint   = [13]
            snowice = [14]

            # to_plot_or_not_2_plot = False
            #plot thresh hist for each obs


            for i, (a, obs) in enumerate(zip(ax.flat, thresh_dict)):
                #edit hists based on applied obs applied as a func of sfc type
                # if   obs == 'WI'      and (k in land or k in water):
                #     to_plot_or_not_2_plot = True
                # elif obs == 'NDVI'    and (k in land or k in water or k in glint):
                #     to_plot_or_not_2_plot = True
                # elif obs == 'NDSI'    and (k in snowice):
                #     to_plot_or_not_2_plot = True
                # elif obs == 'VIS_Ref' and (k in land):
                #     to_plot_or_not_2_plot = True
                # elif obs == 'NIR_Ref' and (k in water):
                #     to_plot_or_not_2_plot = True
                # elif obs == 'Cirrus' or obs == 'SVI':
                #     to_plot_or_not_2_plot = True
                # else:
                #     pass

                # if to_plot_or_not_2_plot:
                if i==0 or i>=3:
                    num_bins = num_bins_other
                    x1, x2   = range_other
                else:
                    num_bins = num_bins_ndxi
                    x1, x2   = range_ndxi

                if k>=12 and k<14:
                    if k==12:
                        col = 'r'
                    elif k==13:
                        col='b'
                    else:
                        col='g'
                    x = np.arange(x1, x2, (x2-x1)/num_bins)
                    if k==12:
                        a.plot(x, temp_thresh[i], label='SID {:02d}'.format(k), c=col)#color[k])
                    else:
                        a.plot(x, temp_thresh[i], label='SID {:02d}'.format(k), c=col, linestyle='dashed')#color[k])

                    if k==13:
                        a.set_title('{} DOY bin {:02d}'.format(obs, DOY_bin))
                        a.legend()

        #only 7 obs so lets turn 8th axis off
        ax[1,3].axis('off')
        home = '/data/keeling/a/vllgsbr2/c/histogram_images_threshold_analysis'
        # plt.savefig('{}/thresh_hist_DOY_bin_{:02d}.pdf'.format(home, DOY_bin), format='pdf')
        print(DOY_bin)
        # plt.legend()
        plt.show()
        # plt.cla()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# n = 100
# number_of_frames = 10
# data = np.random.rand(n, number_of_frames)
#
# def update_hist(num, data):
#     plt.cla()
#     plt.hist(data[num])
#
# fig = plt.figure()
# hist = plt.hist(data[0])
#
# animation = animation.FuncAnimation(fig, update_hist, number_of_frames, fargs=(data, ) )
# plt.show()


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
        thresh_obs_i  = thresh_obs_i.reshape(thresh_shape[0], shape_cosSZA_x_RAZ_x_sfcID)
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
    fill_val = -999.0

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

        #reshape so sfcID is axis 0 and the other axis is everything else flattened
        shape_cosSZA_x_RAZ_x_VZA = int(np.prod(thresh_shape[1:]))
        thresh_obs_i  = thresh_obs_i.reshape((thresh_shape[0], shape_cosSZA_x_RAZ_x_VZA))

        #eliminate fill vals while keeping a vector for each VZA bin
        boxplot_thresh_obs_i = []
        for sfcID_j in range(15):
            thresh_obs_i_sfcID_j = thresh_obs_i[sfcID_j, :]
            filtered_thresh_obs_i_sfcID_j = thresh_obs_i_sfcID_j[thresh_obs_i_sfcID_j > fill_val+2]
            boxplot_thresh_obs_i.append(filtered_thresh_obs_i_sfcID_j)
            # if sfcID_j==13:
            #     print(boxplot_thresh_obs_i[sfcID_j])
        if i==0 or i>=3:
            ymin,ymax = range_other[0], range_other[1]
        else:
            ymin,ymax = range_ndxi[0], range_ndxi[1]

        #a.set_ylim([ymin,ymax])
        a.boxplot(boxplot_thresh_obs_i, notch=False, sym='')
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

    plt.show()

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
        print(thresh_file)
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
            bins = list(hf_group.keys())
            for bin_x in bins:
                if bin_x[-9:-7] == '01':
                    sunglint_bin_data = hf_group[bin_x]
                    print(sunglint_bin_data.shape[0])

def make_obs_hist_by_group(obs):
    '''
    make hists for 100 random bins for an obs from all grouped_obs_and_cm files
    and make them into a movie
    '''

    import matplotlib.pyplot as plt
    from random import sample

    obs_idx_dict = {'WI':1, 'NDVI':2, 'NDSI':3, 'VIS_Ref':4, 'NIR_Ref':5,\
                   'SVI':6, 'Cirrus':7}

    group_home  = config['supporting directories']['combined_group']
    group_home  = '{}/{}/'.format(PTA_path, group_home)
    group_files = [group_home + x for x in os.listdir(group_home)]

    group_hists = []

    # plt.style.use('dark_background')
    # fig, ax=plt.subplots(figsize=(10,10))
    # plt.rcParams['font.size'] = 16
    # container = []

    for i, gf in enumerate(group_files):
        with h5py.File(gf, 'r') as hf_gf:
            bins = list(hf_gf.keys())
            # choose a random subset of 20
            bins_subset = sample(bins, 20)

            # cloud_mask = []
            # obs_x      = []

            for bin in bins_subset:
                data = hf_gf[bin]
                #grab cloud mask and desired observable
                cloud_mask = data[:,0]
                obs_x      = data[:,obs_idx_dict[obs]]

            #divide obs_x by clear and cloudy componants
            obs_x_clear = obs_x[cloud_mask != 0]
            obs_x_cloud = obs_x[cloud_mask == 0]

            #turn these into binned 1d arrays for plotting the histograms
            # min, max = 0, .03
            bin_num  = 128
            # interval = np.abs(max-min)/bin_num
            # bin_params = np.arange(min, max, interval)
            # binned_obs_clear = np.digitize(obs_x_clear, bin_params)
            # binned_obs_cloud = np.digitize(obs_x_cloud, bin_params)

            hist_clear, bin_edges_clear = np.histogram(obs_x_clear,\
                                bins=bin_num, density=False)
            hist_cloud, bin_edges_cloud = np.histogram(obs_x_cloud,\
                                bins=bin_num, density=False)

            #plot
            fig, ax=plt.subplots(figsize=(10,10))
            plt.rcParams['font.size'] = 16

            num_sample_clear = np.sum(hist_clear)
            num_sample_cloud = np.sum(hist_cloud)
            plt.plot(bin_edges_clear[:-1], hist_clear, 'b', label='clear')
            plt.plot(bin_edges_cloud[:-1], hist_cloud, 'r', label='cloudy')
            plt.legend()
            title = 'obs: {}\nbin: {}\n#clear: {}, #cloud: {}'.format(obs, bin, num_sample_clear, num_sample_cloud)
            plt.title(title)
            plt.show()





            # group_hists.append(hf_gf[])
    #         image = ax.imshow(sfc_IDs[:,:,i], cmap=cmap, vmin=0, vmax=11)
    #         title = ax.text(0.5,1.05,'K-Means Cluster Surface ID \nDOY {}/365 Valid previous 8 days'.format(sfc_ID_path[-6:-3]),
    #                         size=plt.rcParams["axes.titlesize"],
    #                         ha="center", transform=ax.transAxes, )
    #
    #         container.append([image, title])
    #
    # ani = animation.ArtistAnimation(fig, container, interval=700, blit=False,
    #                                 repeat=True)
    # ani.save('./{}_hists.mp4'.format(obs))

if __name__ == '__main__':

    # check_neg_SVI_thresh()
    # check_neg_SVI_grouped()
    plot_thresh_hist()
    # plot_thresh_vs_VZA()
    # plot_thresh_vs_sfcID()
    # check_sunglint_thresh()
    # check_sunglint_flag_in_database()
    # check_sunglint_flag_in_grouped_cm_and_obs()
    # make_obs_hist_by_group('Cirrus')








#
