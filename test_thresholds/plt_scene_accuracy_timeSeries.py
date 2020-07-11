import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# 'scene_Accuracy_DOY_bin_{:02d}'.format(DOY_bin)

home = '/data/keeling/a/vllgsbr2/c/old_MAIA_Threshold_dev/LA_PTA_MODIS_Data/try2_database/scene_accuracy'
scene_accur_paths = np.sort(np.array(os.listdir(home)))
scene_accurs = np.zeros((400,300,len(scene_accur_paths)))

plt.style.use('dark_background')
fig, ax=plt.subplots(figsize=(10,10))
cmap = cm.get_cmap('plasma', 20)
plt.rcParams['font.size'] = 16
container = []

# results_ATBD = np.load('./results_so_cool.npz')
# #plot_results(results_ATBD['MCM_accuracy'], results_ATBD['num_samples'])
# scene_Accuracy_DOY_bin_{:02d}.npz

for i, scene_accur_path in enumerate(scene_accur_paths):
    with np.load('{}/{}'.format(home, scene_accur_path)) as npz_scene_accur:
        scene_accurs[:,:,i] = npz_scene_accur['MCM_accuracy']*100
        sample_size = np.sum(npz_scene_accur['num_samples'].flatten())/10**8
        mean_accuracy = np.mean(scene_accurs[:,:,i])
        output = 'DOY Bin {:02d}/45 | mean accuracy {:2.2f} % | num pixels {:1.4} x 10^8'.format(i, mean_accuracy, sample_size)
        print(output)
        image = ax.imshow(scene_accurs[:,:,i], cmap=cmap, vmin=0, vmax=100)
        DOY = (int(scene_accur_path[-6:-4]) + 1)*8
        title = ax.text(0.5,1.05,'Accuracy DOY {}/365\nValid previous 8 days'.format(DOY),
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, )

        container.append([image, title])

ax.set_yticks([])
ax.set_xticks([])
image.cmap.set_under('k')
#cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = plt.colorbar(image,fraction=0.046, pad=0.04)
cbar.set_ticks(np.arange(0,105,5))
cbar.set_ticklabels([str(x) for x in np.arange(0,105,5)])

ani = animation.ArtistAnimation(fig, container, interval=700, blit=False,
                                repeat=True)
ani.save('./MCM_Scene_Accuracy_all_DOY_new_grid.mp4')

plt.show()
