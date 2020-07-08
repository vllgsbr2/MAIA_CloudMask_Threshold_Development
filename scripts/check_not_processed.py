import pandas as pd
import os
import numpy as np
#get txt files
home = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development/scripts/MPI_create_dataset_output'
files = ['{}/{}'.format(home, x) for x in os.listdir(home)]

#master dataframe to store all time stamps processed
master_df = pd.DataFrame(data=None, columns=['time_stamps'], dtype=str)

for f in files:
    # temp_df = pd.read_csv(f, dtype=str, delimiter=',')
    # temp_df.columns = ['num', 'time_stamp', 'phrase']
    temp_t_stamps = []
    with open(f, 'r') as f_current:
        for t_stamp in f_current:
            temp_t_stamps.append(t_stamp.split(', ')[1])
            print(temp_t_stamps)

    master_df['time_stamps'] =  master_df['time_stamps'].append(temp_t_stamps)

processed_files = list(master_df['time_stamps'])
#remove space in front of time stamps
processed_files = [x[1:] for x in processed_files]
#print(master_df)
home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/'
f = 'LAADS_query.2019-10-15T18_07.csv'

laads_query_df = pd.read_csv(home+f)
file_name_column = 'fileUrls from query MOD021KM--61 MOD03--61 MOD35_L2'\
                   '--61 2002-01-01..2019-10-15 x-124.4y39.8 x-112.8y30.7[5]'
all_files = list(laads_query_df[file_name_column])

# check_df = pd.DataFrame()
# check_df.columns = ['time_stamps', 'processed_1_not_processed_0']

check = open('./check_processed.csv', 'w')
check.write('time_stamps,processed_1_not_processed_0\n')

#choose PTA from keeling
PTA_file_path   = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data'

#grab files names for PTA
filename_MOD_02 = np.array(os.listdir(PTA_file_path + '/MOD_02'))

#sort files by time so we can access corresponding files without
#searching in for loop
filename_MOD_02 = np.sort(filename_MOD_02)

#grab time stamp (YYYYDDD.HHMM) to name each group after the granule
#it comes from
time_stamps_downloaded = [x[10:22] for x in filename_MOD_02]

counter = 0
for i in time_stamps_downloaded:
    if i in processed_files:
        print('{} not found #{:0>5d}'.format(i, counter))
        counter+=1

    else:
        check.write('{} not found'.format(i))

check.close()
