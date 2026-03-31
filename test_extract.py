import cv2
import numpy as np
import pprint
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points
import inspect

nusc = NuScenes(version='v1.0-mini', dataroot='c:/Users/ggaka/Downloads/v1.0-mini', verbose=False)

def get_drivable_surface_mask():
    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    cam_token = sample['data']['CAM_FRONT']
    
    # We want to find out how to get map polygons in pixel coordinates
    # Let's see the sourcode of render_map_in_image
    from nuscenes.map_expansion.map_api import NuScenesMap
    
    # init map
    scene_log = nusc.get('log', my_scene['log_token'])
    map_location = scene_log['location']
    nusc_map = NuScenesMap(dataroot='c:/Users/ggaka/Downloads/v1.0-mini', map_name=map_location)

    source_code = inspect.getsource(nusc_map.render_map_in_image)
    print("=== render_map_in_image source ===")
    print(source_code[:1500])
    
get_drivable_surface_mask()
