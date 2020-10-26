def get_MODIS_file_paths(MOD02_txt, MOD03_txt, MOD35_txt):
    import numpy as np
    import os
    with open(MOD02_txt, 'r') as txt_MOD02_files,\
         open(MOD03_txt, 'r') as txt_MOD03_files,\
         open(MOD35_txt, 'r') as txt_MOD35_files :

        #sort paths; path[:-1] is to get rid of hidden new line character
        MOD02_paths = np.sort([path[:-1] for path in txt_MOD02_files])
        MOD03_paths = np.sort([path[:-1] for path in txt_MOD03_files])
        MOD35_paths = np.sort([path[:-1] for path in txt_MOD35_files])

        #grab time stamps of each file
        MOD02_time_stamps = [path[-34:-22] for path in MOD02_paths]
        MOD03_time_stamps = [path[-34:-22] for path in MOD03_paths]
        MOD35_time_stamps = [path[-34:-22] for path in MOD35_paths]

    #if all the time stamps match nothing to do
    if MOD02_time_stamps == MOD03_time_stamps and MOD03_time_stamps == MOD35_time_stamps:
        return MOD02_paths, MOD03_paths, MOD35_paths
    #otherwise find the intercetion of 3 file types and only return those
    else:
        #convert to set to find intersect of all 3
        MOD02_time_stamp_set = set(MOD02_time_stamps)
        MOD03_time_stamp_set = set(MOD03_time_stamps)
        MOD35_time_stamp_set = set(MOD35_time_stamps)

        MOD02_MOD03_time_stamp_intersect = MOD02_time_stamp_set.intersect(MOD03_time_stamp_set)
        MOD02_MOD03_MOD35_time_stamp_intersect = MOD02_MOD03_time_stamp_intersect.intersect(MOD35_time_stamp_set)

        #only save paths in the intersection of 3 sets of file paths
        MOD02_paths = [path for path in MOD02_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]
        MOD03_paths = [path for path in MOD03_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]
        MOD35_paths = [path for path in MOD35_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]

        #Now if the time stamps match we can return themm. Else no return
        MOD02_time_stamps = [path[-34:-22] for path in MOD02_paths]
        MOD03_time_stamps = [path[-34:-22] for path in MOD03_paths]
        MOD35_time_stamps = [path[-34:-22] for path in MOD35_paths]

        if MOD02_time_stamps == MOD03_time_stamps and MOD03_time_stamps == MOD35_time_stamps:
            return MOD02_paths, MOD03_paths, MOD35_paths
        else:
            print('failed due to file time stamps not matching across MOD02/03/35')
            return

def get_MODIS_file_paths_no_list(MOD02_paths, MOD03_paths, MOD35_paths):
    import numpy as np
    import os

    #grab time stamps of each file
    MOD02_time_stamps = [path[-34:-22] for path in MOD02_paths]
    MOD03_time_stamps = [path[-34:-22] for path in MOD03_paths]
    MOD35_time_stamps = [path[-34:-22] for path in MOD35_paths]

    #if all the time stamps match nothing to do
    if MOD02_time_stamps == MOD03_time_stamps and MOD03_time_stamps == MOD35_time_stamps:
        return MOD02_paths, MOD03_paths, MOD35_paths
    #otherwise find the intercetion of 3 file types and only return those
    else:
        #convert to set to find intersect of all 3
        MOD02_time_stamp_set = set(MOD02_time_stamps)
        MOD03_time_stamp_set = set(MOD03_time_stamps)
        MOD35_time_stamp_set = set(MOD35_time_stamps)

        MOD02_MOD03_time_stamp_intersect = MOD02_time_stamp_set.intersect(MOD03_time_stamp_set)
        MOD02_MOD03_MOD35_time_stamp_intersect = MOD02_MOD03_time_stamp_intersect.intersect(MOD35_time_stamp_set)

        #only save paths in the intersection of 3 sets of file paths
        MOD02_paths = [path for path in MOD02_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]
        MOD03_paths = [path for path in MOD03_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]
        MOD35_paths = [path for path in MOD35_paths if path[-34:-22] in MOD02_MOD03_MOD35_time_stamp_intersect]

        #Now if the time stamps match we can return themm. Else no return
        MOD02_time_stamps = [path[-34:-22] for path in MOD02_paths]
        MOD03_time_stamps = [path[-34:-22] for path in MOD03_paths]
        MOD35_time_stamps = [path[-34:-22] for path in MOD35_paths]

        if MOD02_time_stamps == MOD03_time_stamps and MOD03_time_stamps == MOD35_time_stamps:
            return MOD02_paths, MOD03_paths, MOD35_paths
        else:
            print('failed due to file time stamps not matching across MOD02/03/35')
            return


if __name__ == '__main__':
    import configparser
    config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
    config = configparser.ConfigParser()
    config.read(config_home_path+'/test_config.txt')

    PTA      = config['current PTA']['PTA']
    PTA_path = config['PTAs'][PTA]

    MODXX_home = PTA_path + '/' + config['supporting directories']['MODXX_paths_lists']
    MOD02_txt = MODXX_home + '/MOD02_paths.txt'
    MOD03_txt = MODXX_home + '/MOD03_paths.txt'
    MOD35_txt = MODXX_home + '/MOD35_paths.txt'

    filename_MOD_02,\
    filename_MOD_03,\
    filename_MOD_35 = get_MODIS_file_paths(MOD02_txt, MOD03_txt, MOD35_txt)
    n=-1
    print(filename_MOD_02[n], filename_MOD_03[n], filename_MOD_35[n])
