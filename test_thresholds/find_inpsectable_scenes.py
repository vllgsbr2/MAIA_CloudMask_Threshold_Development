def find_good_scenes():
    import os
    import numpy as np
    import h5py
    import configparser

    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config           = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA          = config['current PTA']['PTA']
    PTA_path     = config['PTAs'][PTA]

    data_home          = '{}/{}/'.format(PTA_path, config['supporting directories']['Database'])
    database_files     = os.listdir(datahome)
    database_filepaths = [x + data_home for x in database_files]

    with open('./scenes_worth_inspecting.txt', 'w') as txt_good_scenes:
        for db_file in database_filepaths:
            with h5py.File(db_file, 'r') as hf_file:
                time_stamps = list(hf_file.keys())

                for scene in time_stamps:
                    rad_path = '{}/{}/{}'.format(scene, 'radiance', 'band_1')
                    data     = hf_file[rad_path][()]

                    num_fill_vals    = np.where(data != -999)[0].shape[0]
                    image_shape      = data.shape
                    num_pix_total    = image_shape[0]*image_shape[1]
                    percent_bad_pix  = num_fill_vals/num_pix_total
                    percent_good_pix = 1 - percent_bad_pix

                    if percent_good_pix > 0.1:
                        txt_good_scenes.writeline(scene, percent_good_pix)

if __name__ == '__main__':
    find_good_scenes()
