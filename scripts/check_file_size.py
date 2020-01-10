import pandas as pd
import os.path
from os import path

def check_file_integrity(url, filename, real_file_size, fieldnames, save_path,\
                         directory):
    '''
    Objective:
        Check that file downloaded and has correct size and datafields are not
        corrupt.
    '''

    #make file to save output to
    output_file_path = './diagnostics/LAADS_query.2019-10-15T18_07.csv'
    if path.exists(output_file_path):
        df = pd.read_csv(output_file_path)
    else:
        df         = pd.DataFrame(output_file_path)
        #add column to csv to say if it is corrupt
        df['one_perfect_zero_corrupt']

    #grab some MOD_02/03/35 file based on function arguments
    filename   = save_path + directory + filename
    statinfo   = os.stat(filename)
    downloaded_file_size  = statinfo.st_size

    if url[23:25]=='02':
        if downloaded_file_size == real_file_size :
            df['file_name']

        else:
            output_file.writelines(filename+' failed to download properly'+'\n')

    elif url[23:25]=='03':
        if downloaded_file_size == real_file_size:
            pass
        else:
            output_file.writelines(filename+' failed to download properly'+'\n')

    else:
        if downloaded_file_size == real_file_size:
            pass
        else:
            output_file.writelines(filename+' failed to download properly'+'\n')
