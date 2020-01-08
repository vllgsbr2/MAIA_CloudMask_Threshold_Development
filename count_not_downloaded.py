file_integrity_1 = open('get_PTA_stats_0_58705.txt', 'r')
file_integrity_2 = open('get_PTA_stats_13915_58705.txt', 'r')
corrupt_files = open('corrupt_files.txt', 'w')

for i, x in enumerate(file_integrity_1): 
    if (x[-7:] != 'exists\n') and (x[-11:] != 'downloaded\n'):
        corrupt_files.writelines('{}, {}\n'.format(i, x[29:-28]))
       # print('j')
for i, x in enumerate(file_integrity_2):
    if (x[-7:] != 'exists\n') and (x[-11:] != 'downloaded\n'):
        corrupt_files.writelines('{}, {}\n'.format(i, x[29:-28]))
corrupt_files.close()
