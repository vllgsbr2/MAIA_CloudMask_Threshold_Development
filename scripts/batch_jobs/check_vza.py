import numpy as np
#vzas = np.loadtxt('slurm-3491143.out')
with open('slurm-3491143.out', 'r') as f:
    for x, i in enumerate(f):
        if i=='VZA_08' or i=='VZA_09' or i=='VZA_10':
            print(i, x)
        else:
            print('poop', x)
