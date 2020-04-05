import numpy as np

num_tests    = 7
num_scene_ID = 21

sceneID_configuration = np.ones((num_tests, num_scene_ID))

#just for reference
which_test = {'VIS_Ref':0, 'NIR_Ref':1, 'WI':2, 'NDVI':3,\
              'NDSI':4, 'SVI':5, 'Cirrus':6}
num_land = 12
# land [0,11]
sun_glint                  = num_land + 0
snow                       = num_land + 1
shallow_ocean              = num_land + 2
ocean_lake_coast           = num_land + 3
shallow_inland_water       = num_land + 4
seasonal_inland_water      = num_land + 5
deep_inland_water          = num_land + 6
moderate_continental_ocean = num_land + 7
deep_ocean                 = num_land + 8


#turn off tests over proper sceneID
sceneID_configuration = \
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],\
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],\
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],\
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],\
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20]

#VIS_Ref; off over water/snow/glint
#NIR_Ref; off over land/glint/snow
#WI; off over glint/snow
#NDVI; off over snow
#NDSI; off over water/glint/snow/land
#SVI; off no where
#Cirrus; off no where

np.savez('./sceneID_configuration.npz', x=sceneID_configuration)
