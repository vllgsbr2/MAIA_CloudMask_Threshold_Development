import h5py
import numpy as np
from rgb_enhancement import get_enhanced_RGB
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpl_c
import matplotlib
import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import configparser
from distribute_cores import distribute_processes
matplotlib.use('Agg')
# import mpi4py.MPI as MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# for r in range(size):
#     if rank==r:

#read in config file
config_home_path = '/data/keeling/a/vllgsbr2/c/MAIA_thresh_dev/MAIA_CloudMask_Threshold_Development'
config = configparser.ConfigParser()
config.read(config_home_path+'/test_config.txt')

PTA          = config['current PTA']['PTA']
PTA_path     = config['PTAs'][PTA]

#grab output files
MCM_output_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/MCM_Output/numKmeansSID_16/'
time_stamps     = np.sort(os.listdir(MCM_output_home))

#grab input files
MCM_input_home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/MCM_Input/'
test_data_JPL_paths = os.listdir(MCM_input_home)
time_stamps         = [x[14:26] for x in test_data_JPL_paths]
test_data_JPL_paths = [MCM_input_home + x for x in test_data_JPL_paths]

# #assign subset of files to current rank
# num_processes = len(test_data_JPL_paths)
# start, stop   = distribute_processes(size, num_processes)
# start, stop   = start[rank], stop[rank]
# time_stamps, test_data_JPL_paths = time_stamps[start:stop], test_data_JPL_paths[start:stop]

#create figure instance
left   = 0.005
right  = 0.985
bottom = 0.110
top    = 0.880
wspace = 0.060
hspace = 0.090

f, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 12))
f.subplots_adjust(bottom=bottom, right=right, top=top, wspace=wspace,\
                  hspace=hspace, left=left)

# thesisCaseTimestamps = ['2018362.1800',\
#                         '2017014.1805',\
#                         '2018357.1920',\
#                         '2015297.1805',\
#                         '2018355.1930',\
#                         '2014210.1830',\
#                         '2018354.1850',\
#                         '2018350.1915',\
#                         '2017008.1845',\
#                         '2015291.1845',\
#                         '2018349.1830',\
#                         '2017013.1900',\
#                         '2017011.1915',\
#                         '2014156.1910',\
#                         '2014135.1850',\
#                         '2013317.1845',\
#                         '2014133.1900',\
#                         '2014126.1855',\
#                         '2013219.1855',\
#                         '2012224.1900',\
#                         '2011207.1850',\
#                         '2011198.1855',\
#                         '2012233.1855',\
#                         '2011273.1835',\
#                         '2011262.1855',\
#                         '2011210.1920' ]

# thesisCaseTimestamps = ['2009244.1830']

thesisCaseTimestamps = ['2015291.1845', '2018349.1830',\
                        '2014133.1900', '2017350.1805']

#make new cmap 0-15 cmap continuous/ C red/W blue/SGW yellow/SI white
ocean = cm.get_cmap('ocean', 20)
newcolors = ocean(np.linspace(0, 1, 20))
newcolors[16, :] = mpl_c.to_rgba('red')
newcolors[17, :] = mpl_c.to_rgba('cyan')
newcolors[18, :] = mpl_c.to_rgba('yellow')
newcolors[19, :] = mpl_c.to_rgba('white')
newcmp = ListedColormap(newcolors)



for time_stamp, test_data_JPL_path in zip(time_stamps, test_data_JPL_paths):
    #just produce for cases above
    if time_stamp not in thesisCaseTimestamps:
        continue
    # else:
    #     print(time_stamp)
    output_file_path = MCM_output_home + time_stamp + '/MCM_Output.h5'
    # #skip files already processed
    home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/PTAs/LosAngeles/results/thesisCasePlots/'
    save_path = home + time_stamp +'.pdf'
    # if os.path.exists(save_path):
    #     if os.path.getsize(save_path) > 0:
    #         continue
    with h5py.File(output_file_path, 'r') as hf_MCM_output:
        DTT = hf_MCM_output['cloud_mask_output/DTT'][()]
        MCM = hf_MCM_output['cloud_mask_output/final_cloud_mask'][()]
        SID = hf_MCM_output['Ancillary/scene_type_identifier'][()]

        #get RGB
        R_red = hf_MCM_output['Reflectance/band_06'][()]
        R_grn = hf_MCM_output['Reflectance/band_05'][()]
        R_blu = hf_MCM_output['Reflectance/band_04'][()]

        #if no data anywhere just goto next file
        if np.all(R_red==-999):
            continue
        RGB = np.dstack((R_red, R_grn, R_blu))
        RGB[RGB==-999] = 0
        RGB = get_enhanced_RGB(RGB)

    #grab mod35 cm from input file
    with h5py.File(test_data_JPL_path, 'r') as hf_output:
        mod35cm = hf_output['MOD35_cloud_mask'][()]

    #plot
    #DTT_WI, DTT_NDVI, DTT_NDSI, DTT_VIS_Ref, DTT_NIR_Ref, DTT_SVI, DTT_Cirrus
    # obs_namelist = ['WI', 'NDVI', 'NDSI', '0.65µm BRF', '0.86µm BRF', 'SVI', 'Cirrus']
    obs_namelist = ['0.65µm BRF', '0.86µm BRF', 'Cirrus', 'WI', 'SVI', 'NDVI', 'NDSI']
    obs_idxlist  = [3, 4, 6, 0, 5, 1, 2]
    # In your thesis, you may want to try the following layout
    # Top row:  RGB, MCM, MOD35, SID
    # Middle row: 0.65, 0.85, Cirrus, WI,
    # Bottom row:  SVI, NDVI, NDSI
    #
    # In the caption, you need to specify the granule name and date/time.

    for i, a in enumerate(ax.flat):

        #plot BRF/MOD35/MCM/SID
        if i==0:
            im = a.imshow(RGB, vmin=0)
            a.set_title('RGB')
            im.cmap.set_under('r')
        elif i==1:
            im = a.imshow(MCM, vmin=0, vmax=1, cmap='binary')
            a.set_title('MCM')
            im.cmap.set_under('r')
            im.cmap.set_over('r')

        elif i==2:
            im_mod35 = a.set_title('MOD35')
            cmap = ListedColormap(['white', 'green', 'blue','black'])
            norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap.N)
            im_mod35 = a.imshow(mod35cm, vmin=0, cmap=cmap, norm=norm)
            cax = f.add_axes([0.77, 0.11, 0.012, 0.24])
            cbar = f.colorbar(im_mod35, cax=cax, orientation='vertical')
            cbar.set_ticks([0.5,1.5,2.5,3.5])
            cbar.set_ticklabels(['CD', 'UCR', \
                                 'PCR', 'CR'])
            im_mod35.cmap.set_under('r')


        elif i==3:
            cmap = newcmp#cm.get_cmap('ocean', 20)
            im_SID = a.imshow(SID, vmin=0, vmax=20, cmap=cmap)
            print(SID.max())
            a.set_title('SID')
            cax = f.add_axes([0.83, 0.11, 0.012, 0.24])
            cbar = f.colorbar(im_SID, cax=cax, orientation='vertical')
            cbar.set_ticks(np.arange(0.5,20.5))
            SID_cbar_labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','C', 'W', 'SGW', 'SI']
            cbar.set_ticklabels(SID_cbar_labels)
            im_SID.cmap.set_under('r')


        #plot DTT first
        elif i > 3 and i < 11:
            im_DTT = a.imshow(DTT[:,:,obs_idxlist[i-4]], vmin=-101, vmax=101, cmap='bwr')
            a.set_title(obs_namelist[i-4])
            im.cmap.set_under('k')
            cax = f.add_axes([0.89, 0.11, 0.012, 0.24])
            cbar = f.colorbar(im_DTT, cax=cax, orientation='vertical')
            im_DTT.cmap.set_under('k')


        #turn off unused axes
        elif i >= 11:
            a.axis('off')
        else:
            pass
        #turn off ticks
        a.set_xticks([])
        a.set_yticks([])

    f.savefig(save_path+'.png', dpi=300, format='png')
    print(time_stamp)
    # plt.show()
    for a in ax.flat:
        a.clear()
