import os
import cv2
import numpy as np
import shapely.geometry
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.utils.geometry_utils import view_points, box_in_image
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion
except ImportError:
    print("Please install required packages:")
    print("pip install nuscenes-devkit opencv-python shapely tqdm")
    exit(1)

# Configuration
DATAROOT = '/content/drive/MyDrive/nuscenes_project'
VERSION = 'v1.0-trainval'
OUTPUT_DIR = '/content/drive/MyDrive/DrivableSpaceDataset'
TARGET_SIZE = (512, 256) # (Width, Height)

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'completed_scenes.txt')

maps_cache = {}

def get_map(location):
    if location not in maps_cache:
        maps_cache[location] = NuScenesMap(dataroot=DATAROOT, map_name=location)
    return maps_cache[location]

def load_completed_scenes():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return set(f.read().splitlines())
    return set()

def mark_scene_completed(scene_name):
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(scene_name + '\n')

def create_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

def transform_global_to_camera(point_global, ego_pose, calibrated_sensor):
    """ Transforms a point from global coordinates to camera coordinates. """
    # 1. Global -> Ego
    ego_quat = Quaternion(ego_pose['rotation'])
    pt_ego = point_global - np.array(ego_pose['translation'])
    pt_ego = ego_quat.inverse.rotate(pt_ego)
    
    # 2. Ego -> Camera/Sensor
    cs_quat = Quaternion(calibrated_sensor['rotation'])
    pt_cam = pt_ego - np.array(calibrated_sensor['translation'])
    pt_cam = cs_quat.inverse.rotate(pt_cam)
    return pt_cam

def process_scene(nusc, scene, split):
    # Initialize Map for this scene's log
    log = nusc.get('log', scene['log_token'])
    try:
        nusc_map = get_map(log['location'])
    except FileNotFoundError:
        print(f"ERROR: Map expansion JSONs missing for {log['location']}.")
        print("Please download 'nuScenes-map-expansion-v1.3.zip' from nuScenes and extract into v1.0-mini/maps/expansion/")
        return

    # Iterating over all samples in the scene
    curr_sample_token = scene['first_sample_token']
    
    while curr_sample_token != '':
        sample = nusc.get('sample', curr_sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        
        # Load Original Image
        img_path = os.path.join(DATAROOT, cam_data['filename'])
        if not os.path.exists(img_path):
            curr_sample_token = sample['next']
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Corrupted or missing image: {img_path}")
            curr_sample_token = sample['next']
            continue
            
        orig_h, orig_w = img.shape[:2]
        
        # We need ego pose and camera calibration to project map polygons
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsics = np.array(cs['camera_intrinsic'])
        
        # Create empty mask (same size as original image)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # 1. Project Drivable Surface Polygons
        ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]
        
        # Fetch polygons within a radius of the ego vehicle to speed up
        records = nusc_map.get_records_in_radius(ego_x, ego_y, 100, ['drivable_area'])
        drivable_tokens = records['drivable_area']
        
        for token in drivable_tokens:
            record = nusc_map.get('drivable_area', token)
            for polygon_token in record['polygon_tokens']:
                polygon = nusc_map.extract_polygon(polygon_token)
                # Outer contour
                pts = np.array(polygon.exterior.coords)
                
                # We need Z to be at general ground level (ego_pose z) to project correctly
                # Map polygons are 2D, we assume they lie on the floor of the ego pose
                z_floor = ego_pose['translation'][2] 
                
                pts_3d = np.hstack((pts, np.full((len(pts), 1), z_floor)))
                
                # Transform to camera coords
                pts_cam = np.array([transform_global_to_camera(pt, ego_pose, cs) for pt in pts_3d]).T
                
                # Filter points behind camera (z < 0.1) to avoid crazy projections 
                # (Simple clippping approximation)
                # Fix: Strip invalid point projections, explicitly clip mask to screen boundaries
                valid = pts_cam[2, :] > 0.1
                if valid.sum() < 3:
                    continue
                    
                pts_img = view_points(pts_cam[:, valid], intrinsics, normalize=True)
                pts_img = pts_img[:2, :].T
                pts_img[:, 0] = np.clip(pts_img[:, 0], 0, orig_w - 1)
                pts_img[:, 1] = np.clip(pts_img[:, 1], 0, orig_h - 1)
                pts_img = pts_img.astype(np.int32)
                cv2.fillPoly(mask, [pts_img], 255)

        # 2. Subtract Dynamic Objects (Vehicles, Pedestrians)
        _, boxes, _ = nusc.get_sample_data(cam_token)
        for box in boxes:
            if 'vehicle' in box.name or 'human' in box.name or 'movable_object' in box.name:
                # Get the 8 corners of the 3D box
                corners = box.corners() # 3x8
                # Project to image
                corners_img = view_points(corners, intrinsics, normalize=True)
                # Use robust nuScenes native box projection filter
                if not box_in_image(box, intrinsics, (orig_w, orig_h)):
                    continue
                
                corners_img = corners_img[:2, :].T.astype(np.int32)
                # Use convex hull to mask out the entire box area
                hull = cv2.convexHull(corners_img)
                cv2.fillPoly(mask, [hull], 0)

        # Resize Image and Mask to Target Size (512x256)
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is strictly binary [0, 1] for training
        mask_binary = (mask_resized > 127).astype(np.uint8)

        # Save outputs
        base_name = f"{scene['name']}_{curr_sample_token}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'images', f"{base_name}.jpg"), img_resized)
        cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'masks', f"{base_name}.png"), mask_binary * 255)
        
        curr_sample_token = sample['next']


def main():
    print("Loading NuScenes Metadata...")
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    
    # 1. Deterministic Split Formulation
    scenes = sorted(nusc.scene, key=lambda x: x['name'])
    
    # Create dynamic 80/20 train/val scene split
    split_idx = int(len(scenes) * 0.8)
    train_scenes = scenes[:split_idx]
    val_scenes = scenes[split_idx:]
    
    print(f"Total Scenes: {len(scenes)}")
    print(f"Train Scenes: {[s['name'] for s in train_scenes]}")
    print(f"Val Scenes: {[s['name'] for s in val_scenes]}")
    
    create_directories()
    
    completed = load_completed_scenes()
    print(f"Already completed: {len(completed)} scenes")
    
    print("Processing Train Split...")
    for scene in tqdm(train_scenes):
        if scene['name'] in completed:
            continue
        process_scene(nusc, scene, 'train')
        mark_scene_completed(scene['name'])
        
    print("Processing Val Split...")
    for scene in tqdm(val_scenes):
        if scene['name'] in completed:
            continue
        process_scene(nusc, scene, 'val')
        mark_scene_completed(scene['name'])
        
    print(f"Dataset generation complete. Saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
