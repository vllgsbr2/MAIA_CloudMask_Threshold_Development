import numpy as np

num_tests    = 7
num_scene_ID = 21

sceneID_configuration = np.zeros((num_tests, num_scene_ID))

which_test = {'VIS_Ref':0, 'NIR_Ref':1, 'WI':2, 'NDVI':3,\
              'NDSI':4, 'SVI':5, 'Cirrus':6}

#land [0,11]
# sun_glint                  12+0
# snow                       12+1
# shallow_ocean              12+2
# ocean_lake_coast           12+3
# shallow_inland_water       12+4
# seasonal_inland_water      12+5
# deep_inland_water          12+6
# moderate_continental_ocean 12+7
# deep_ocean                 12+8

sceneID_configuration[which_test['VIS_Ref']] = np.array([])
