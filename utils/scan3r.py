import os.path as osp
import numpy as np
import json
from glob import glob
from plyfile import PlyData, PlyElement
import cv2
import pickle

def get_scan_ids(dirname, split):
    filepath = osp.join(dirname, '{}_scans.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def read_labels(plydata):
    data = plydata.metadata['_ply_raw']['vertex']['data']
    try:
        labels = data['objectId']
    except:
        labels = data['label']
    return labels

def load_intrinsics(data_dir, scan_id, type='color'):
    '''
    Load 3RScan intrinsic information
    '''
    info_path = osp.join(data_dir, scan_id, 'sequence', '_info.txt')

    width_search_string = 'm_colorWidth' if type == 'color' else 'm_depthWidth'
    height_search_string = 'm_colorHeight' if type == 'color' else 'm_depthHeight'
    calibration_search_string = 'm_calibrationColorIntrinsic' if type == 'color' else 'm_calibrationDepthIntrinsic'

    with open(info_path) as f:
        lines = f.readlines()
    
    for line in lines:
        if line.find(height_search_string) >= 0:
            intrinsic_height = line[line.find("= ") + 2 :]
        
        elif line.find(width_search_string) >= 0:
            intrinsic_width = line[line.find("= ") + 2 :]
        
        elif line.find(calibration_search_string) >= 0:
            intrinsic_mat = line[line.find("= ") + 2 :].split(" ")

            intrinsic_fx = intrinsic_mat[0]
            intrinsic_cx = intrinsic_mat[2]
            intrinsic_fy = intrinsic_mat[5]
            intrinsic_cy = intrinsic_mat[6]

            intrinsic_mat = np.array([[intrinsic_fx, 0, intrinsic_cx],
                                    [0, intrinsic_fy, intrinsic_cy],
                                    [0, 0, 1]])
            intrinsic_mat = intrinsic_mat.astype(np.float32)
    intrinsics = {'width' : float(intrinsic_width), 'height' : float(intrinsic_height), 
                  'intrinsic_mat' : intrinsic_mat}
    
    return intrinsics

def load_ply_data(data_dir, scan_id, label_file_name):
    filename_in = osp.join(data_dir, scan_id, label_file_name)
    file = open(filename_in, 'rb')
    ply_data = PlyData.read(file)
    file.close()
    return ply_data

def load_pose(data_dir, scan_id, frame_id):
    pose_path = osp.join(data_dir, scan_id, 'sequence', 'frame-{}.pose.txt'.format(frame_id))
    pose = np.genfromtxt(pose_path)
    return pose

def load_all_poses(data_dir, scan_id, frame_idxs):
    frame_poses = []
    for frame_idx in frame_idxs:
        frame_pose = load_pose(data_dir, scan_id, frame_idx)
        frame_poses.append(frame_pose)
    frame_poses = np.array(frame_poses)

    return frame_poses

def load_frame_idxs(data_dir, scan_id, skip=None):
    num_frames = len(glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg')))

    if skip is None:
        frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames)]
    else:
        frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames, skip)]
    return frame_idxs

def load_frame_paths(data_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    img_folder = osp.join(data_dir, "scenes", scan_id, 'sequence')
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = "frame-{}.color.jpg".format(frame_idx)
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths

def load_depth_paths(data_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    img_folder = osp.join(data_dir, "scenes", scan_id, 'sequence')
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = "frame-{}.depth.pgm".format(frame_idx)
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths

def load_patch_feature_scans(data_root_dir, feature_folder, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_root_dir, "scenes"), scan_id, skip)
    features_file = osp.join(osp.join(data_root_dir, "files"), feature_folder, scan_id+".pkl")
    with open(features_file, 'rb') as handle:
        features_scan = pickle.load(handle)
    features_scan_step = {}
    for frame_idx in frame_idxs:
        features_scan_step[frame_idx] = features_scan[frame_idx]
    return features_scan_step

def load_pkl_data(filename):
    with open(filename, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict


def load_gt_2D_anno(data_root_dir, scan_id, skip=None):
    anno_imgs = {}
    frame_idxs = load_frame_idxs(osp.join(data_root_dir, "scenes"), scan_id, skip)
    anno_folder = osp.join(data_root_dir, "files", 'gt_projection/obj_id', scan_id)
    for frame_idx in frame_idxs:
        anno_img_file = osp.join(anno_folder, "frame-{}.jpg".format(frame_idx))
        anno_img = cv2.imread(anno_img_file, cv2.IMREAD_UNCHANGED)
        anno_imgs[frame_idx] = anno_img
    return anno_imgs
    

def read_transform_mat(filename):
    rescan2ref = {}
    with open(filename , "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
    return rescan2ref

def load_plydata_npy(file_path, obj_ids = None, return_ply_data = False):
    ply_data = np.load(file_path)
    points =  np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))

    if obj_ids is not None:
        if type(obj_ids) == np.ndarray:
            obj_ids_pc = ply_data['objectId']
            obj_ids_pc_mask = np.isin(obj_ids_pc, obj_ids)
            points = points[np.where(obj_ids_pc_mask == True)[0]]
        else:
            obj_ids_pc = ply_data['objectId']
            points = points[np.where(obj_ids_pc == obj_ids)[0]]
    
    if return_ply_data: return points, ply_data
    else: return points
    
def sampleCandidateScenesForEachScan(scan_id, scan_ids, 
                                        refscans2scans, scans2refscans , num_scenes):
    import random
    scans_same_scene = refscans2scans[scans2refscans[scan_id]]
    # sample other scenes
    sample_candidate_scans = [scan for scan in scan_ids if scan not in scans_same_scene]
    if num_scenes < 0:
        return sample_candidate_scans
    elif num_scenes <= len(sample_candidate_scans):
        return random.sample(sample_candidate_scans, num_scenes)
    else:
        return sample_candidate_scans

def find_cam_centers(frame_idxs, frame_poses):
    cam_centers = []

    for idx in range(len(frame_idxs)):
        cam_2_world_pose = frame_poses[idx]
        frame_pose = np.linalg.inv(cam_2_world_pose) # world To Cam
        frame_rot = frame_pose[:3, :3]
        frame_trans = frame_pose[:3, 3] * 1000.0
        cam_center = -np.matmul(np.transpose(frame_rot), frame_trans)
        cam_centers.append(cam_center / 1000.0)

    cam_centers = np.array(cam_centers).reshape((-1, 3))
    return cam_centers

def create_ply_data_predicted(ply_data, visible_pts_idx):
    x = ply_data['vertex']['x'][visible_pts_idx]
    y = ply_data['vertex']['y'][visible_pts_idx]
    z = ply_data['vertex']['z'][visible_pts_idx]
    object_id = ply_data['vertex']['label'][visible_pts_idx]

    vertices = np.empty(len(visible_pts_idx), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('objectId', 'h')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['objectId'] = object_id.astype('h')

    return vertices, object_id

def create_ply_data(ply_data, visible_pts_idx):
    x = ply_data['vertex']['x'][visible_pts_idx]
    y = ply_data['vertex']['y'][visible_pts_idx]
    z = ply_data['vertex']['z'][visible_pts_idx]
    red = ply_data['vertex']['red'][visible_pts_idx]
    green = ply_data['vertex']['green'][visible_pts_idx]
    blue = ply_data['vertex']['blue'][visible_pts_idx]
    object_id = ply_data['vertex']['objectId'][visible_pts_idx]
    global_id = ply_data['vertex']['globalId'][visible_pts_idx]
    nyu40_id = ply_data['vertex']['NYU40'][visible_pts_idx]
    eigen13_id = ply_data['vertex']['Eigen13'][visible_pts_idx]
    rio27_id = ply_data['vertex']['RIO27'][visible_pts_idx]

    vertices = np.empty(len(visible_pts_idx), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                                     ('objectId', 'h'), ('globalId', 'h'), ('NYU40', 'u1'), ('Eigen13', 'u1'), ('RIO27', 'u1')])
    
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')
    vertices['objectId'] = object_id.astype('h')
    vertices['globalId'] = global_id.astype('h')
    vertices['NYU40'] = nyu40_id.astype('u1')
    vertices['Eigen13'] = eigen13_id.astype('u1')
    vertices['RIO27'] = rio27_id.astype('u1')

    return vertices, object_id

def load_depth_map(depth_file, scale):
    depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_map = depth_map.astype(np.float32) / scale
    return depth_map

def load_scan_depth_map(data_dir, scan_id, scale, step=None):
    depth_paths = load_depth_paths(data_dir, scan_id, step)
    depth_maps = {}
    for frame_idx, depth_path in depth_paths.items():
        depth_map = load_depth_map(depth_path, scale)
        depth_maps[frame_idx] = depth_map
    return depth_maps

def depthmap2pc(depth_map, intrinsic, depth_range):

    # get intrinsic parameters
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    
    # get image size
    height = depth_map.shape[0]
    width = depth_map.shape[1]
    
    # get pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    
    # get depth values
    depth = depth_map.flatten()
    
    # get 3D coordinates
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth
    
    # get valid points
    valid = np.where((depth > depth_range[0]) & (depth < depth_range[1]))
    x3 = x3[valid]
    y3 = y3[valid]
    z3 = z3[valid]
    
    # get point cloud
    pc = np.stack([x3, y3, z3, np.ones_like(x3)]).transpose((1, 0))
    
    # shift from cam coords to KITTI style (x-forward, y-left, z-up)
    pc_kitti = pc[:, (2, 0, 1, 3)]
    pc_kitti[:, 1] = -pc_kitti[:, 1]
    pc_kitti[:, 2] = -pc_kitti[:, 2]
    
    pc_valid = pc_kitti[ np.isfinite(pc_kitti).all(axis=1) ]
    pc_valid = pc_valid[ np.isnan(pc_valid).any(axis=1) == False ]
    return pc_valid

def load_scan_depth_pcs(data_dir, scan_id, scale, intrinsic, depth_range, step=1):
    depth_paths = load_depth_paths(data_dir, scan_id, step)
    depth_pcs = {}
    for frame_idx, depth_path in depth_paths.items():
        depth_map = load_depth_map(depth_path, scale)
        pcs = depthmap2pc(depth_map, intrinsic, depth_range)
        pcs_valid = pcs[ np.isfinite(pcs).all(axis=1) ]
        pcs_valid = pcs_valid[ np.isnan(pcs_valid).any(axis=1) == False ]
        if pcs_valid.shape[0] < 1000:
            print('No valid points in frame {}'.format(frame_idx))
            continue
        depth_pcs[frame_idx] = pcs_valid
    return depth_pcs

def load_scan_pcs(data_dir, scan_id, transform_rescan2ref, ref_coord = False, color=False):
    ply_data_npy_file = osp.join(data_dir, scan_id, 'data.npy')
    ply_data = np.load(ply_data_npy_file)
    if color:
        points = np.stack(
            [ply_data['x'], 
             ply_data['y'], 
             ply_data['z'], 
             np.ones_like(ply_data['x']),
             ply_data['red'], 
             ply_data['green'], 
             ply_data['blue'],]
            ).transpose((1, 0))
    else:
        points = np.stack(
            [ply_data['x'], 
             ply_data['y'], 
             ply_data['z'], 
             np.ones_like(ply_data['x']) ]
            ).transpose((1, 0))
    # the loaded point cloud is in the coordinate of the reference scan
    if not ref_coord and scan_id in transform_rescan2ref:
        T_ref2rescan = np.linalg.inv(transform_rescan2ref[scan_id]).astype(np.float32)
        # assert( abs(np.linalg.det(T_ref2rescan) - 1) < 1e-5)
        # T_ref2rescan = np.eye(4, dtype=np.float32)
        points[:,:4] = np.asarray(points[:,:4]@ T_ref2rescan).astype(np.float32)
    
    return points

def createRangeImage(pcs, colors, center, fov_up, fov_down, proj_W, proj_H, range):
    # pcs: (N, 4)
    # colors: (N, 3)
    # center: (3,)
    # fov_up: float in degree
    # fov_down: float in degree
    # range: [float, float] - [min, max]
    # return: (H, W, 3), (H, W, 3) - depth map and color map
    
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    current_vertex = pcs - center
    min_range = range[0]
    max_range = range[1]
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    is_in_range = (depth > min_range) & (depth < max_range)
    current_vertex = current_vertex[is_in_range]  # get rid of out of range points
    colors = colors[is_in_range]
    depth = depth[is_in_range]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    colors = colors[order, :]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W, 3), 0,
                        dtype=np.uint8)  # [H,W] range (-1 is no data)
    proj_color = np.full((proj_H, proj_W, 3), 0,
                        dtype=np.uint8)  # [H,W] index (0 is no data)

    depth_uint8 = (depth / max_range * 255).astype(np.uint8)
    proj_range[proj_y, proj_x] = np.stack( (depth_uint8,)*3, axis=-1)
    proj_color[proj_y, proj_x] = colors
    return proj_range, proj_color

def loadScanMeshRange(mesh_file, fov_up, fov_down, range_min, range_max, range_H, range_W):
    import open3d as o3d
    # load scene
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # generate rays 
    rays = np.zeros((range_H * range_W, 6), dtype=np.float32)
    degree_to_radian =  np.pi/180.0
    yaws = np.linspace(-np.pi, np.pi, range_W, endpoint=False)
    pitchs = np.linspace(fov_up*degree_to_radian,  fov_down * degree_to_radian, range_H, endpoint=False)
    coord_x_W = np.cos(yaws)
    coord_y_W = np.sin(yaws)
    coord_z_H = np.sin(pitchs)
    coord_index_xy, coord_index_z = np.meshgrid(np.arange(range_W), np.arange(range_H))
    coord_x_W = coord_x_W[coord_index_xy.reshape(-1)]
    coord_y_W = coord_y_W[coord_index_xy.reshape(-1)]
    coord_z_H = coord_z_H[coord_index_z.reshape(-1)]
    coords = np.stack((coord_x_W, coord_y_W, coord_z_H), axis=1)
    coords = coords / np.linalg.norm(coords, axis=1, ord=2, keepdims=True)
    rays[:, 3:] = coords
    center = np.mean(np.asarray(mesh.vertices), axis=0)
    rays[:, :3] = center
    # raycasting
    ans = scene.cast_rays(rays)
    trian_ids = ans['primitive_ids'].numpy()
    depth = ans['t_hit'].numpy()
    valid_mask = np.logical_and(depth > range_min, depth < range_max)
    depth_valid_norm = (depth[valid_mask] * 255.0 / range_max).astype(np.uint8)
    # generate image
    mesh_triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)*255.0
    trians_valid = mesh_triangles[trian_ids[valid_mask]]
    points_ids_valid = trians_valid[:, 0]
    colors_valid = colors[points_ids_valid]
    hit_points_ids_valid = np.arange(len(trian_ids))[valid_mask]
    color_map = np.zeros((range_H, range_W, 3), dtype=np.uint8)
    depth_map = np.zeros((range_H, range_W, 3), dtype=np.uint8)
    coord_w = coord_index_xy.reshape(-1)[valid_mask]
    coord_h = coord_index_z.reshape(-1)[valid_mask]
    color_map[coord_h, coord_w] = colors_valid
    depth_map[coord_h, coord_w] = depth_valid_norm.reshape(-1, 1)
    
    return depth_map, color_map
    