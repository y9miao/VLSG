import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

import cv2
import open3d as o3d
# import open3d.visualization.rendering as rendering
from tqdm import tqdm
import sys
sys.path.append('..')

from utils import common, scan3r

class Scan3RIMGProjector():
    def __init__(self, data_root_dir, split, use_rescan=False):
        self.split = split
        self.use_rescan = use_rescan
        self.data_root_dir = data_root_dir
        
        scan_dirname = ''
        self.scans_dir = osp.join(data_root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        self.scenes_config_file = osp.join(self.scans_dir, 'files', '3RScan.json')
        self.scenes_configs = common.load_json(self.scenes_config_file)
        self.objs_config_file = osp.join(self.scans_dir, 'files', 'objects.json')
        self.objs_configs = common.load_json(self.objs_config_file)
        self.scan_ids = []
        
        # get scans
        for scan_data in self.scenes_configs:
            if scan_data['type'] == self.split:
                self.scan_ids.append(scan_data['reference'])
                if self.use_rescan:
                    rescan_ids = [scan['reference'] for scan in scan_data['scans']]
                    self.scan_ids += rescan_ids
                    
        self.scan_ids.sort()
        
        # get save dir 
        self.save_dir = osp.join(self.scans_dir, 'files', 'gt_projection')
        self.save_color_dir = osp.join(self.save_dir, 'color')
        self.save_obj_dir = osp.join(self.save_dir, 'obj_id')
        common.ensure_dir(self.save_dir)
        common.ensure_dir(self.save_color_dir)
        common.ensure_dir(self.save_obj_dir)
      
    def __len__(self):
        return len(self.scan_ids)

    def project(self, scan_idx, step = 1):
        # get related files
        scan_id = self.scan_ids[scan_idx]
        mesh_file = osp.join(self.scans_scenes_dir, scan_id, "labels.instances.annotated.v2.ply")
        imgs_folder = osp.join(self.scans_scenes_dir, scan_id, "sequence")
        intrinsic_file = osp.join(imgs_folder, "_info.txt")
        
        # get img info and camera intrinsics 
        camera_info = scan3r.load_intrinsics(self.scans_scenes_dir, scan_id)
        intrinsics = camera_info['intrinsic_mat']
        img_width = int(camera_info['width'])
        img_height = int(camera_info['height'])
        
        # load labels
        plydata_npy = np.load(osp.join(self.scans_scenes_dir, scan_id, "data.npy"))
        obj_labels = plydata_npy['objectId']
    
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh_triangles = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)*255.0
        colors = colors.round()
        num_triangles = mesh_triangles.shape[0]
        
        # load scene 
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh_triangles = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)*255.0
        colors = colors.round()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        
        # get frame_indexes
        frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id)
        poses = scan3r.load_all_poses(self.scans_scenes_dir, scan_id, frame_idxs)
        
        # project 3D model
        obj_id_imgs = {}
        color_imgs = {}
        for idx in range(0, len(poses), step):
            frame_idx = frame_idxs[idx]
            img_pose = poses[idx]
            img_pose_inv = np.linalg.inv(img_pose)
            color_map, obj_id_map = self.segmentResult(
                scene, intrinsics, img_pose_inv, img_width, img_height, 
                mesh_triangles, num_triangles, colors, obj_labels
            )
            obj_id_imgs[frame_idx] = obj_id_map
            color_imgs[frame_idx] = color_map
            
        # save 
        save_scan_color_dir = osp.join(self.save_color_dir, scan_id)
        save_scan_obj_dir = osp.join(self.save_obj_dir, scan_id)
        common.ensure_dir(save_scan_color_dir)
        common.ensure_dir(save_scan_obj_dir)
        
        for frame_idx in obj_id_imgs:
            obj_id_img = obj_id_imgs[frame_idx]
            color_img = color_imgs[frame_idx]
            
            img_name = "frame-"+str(frame_idx)+".jpg"
            obj_id_img_file = osp.join(save_scan_obj_dir, img_name)
            color_img_file = osp.join(save_scan_color_dir, img_name)
            cv2.imwrite(obj_id_img_file, obj_id_img)
            cv2.imwrite(color_img_file, color_img)
            
    
    def segmentResult(self, scene, intrinsics, extrinsics, width, height,
                      mesh_triangles, num_triangles, colors, obj_ids):
        
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix = intrinsics.astype(np.float64),
            extrinsic_matrix = extrinsics.astype(np.float64),
            width_px = width, height_px = height
        )
        
        ans = scene.cast_rays(rays)
        hit_triangles_ids = ans['primitive_ids'].numpy()
        hit_triangles_ids_valid_masks = (hit_triangles_ids<num_triangles)
        hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
        hit_triangles_valid = mesh_triangles[hit_triangles_ids_valid]
        hit_points_ids_valid = hit_triangles_valid[:,0]
        
        color_map = np.zeros((height,width,3), dtype=np.uint8)
        obj_id_map = np.zeros((height,width), dtype=np.uint8)
        color_map[hit_triangles_ids_valid_masks] = colors[hit_points_ids_valid]
        obj_id_map[hit_triangles_ids_valid_masks] = obj_ids[hit_points_ids_valid]
        
        return color_map, obj_id_map
        
if __name__ == '__main__':

    data_root_dir = "/home/yang/big_ssd/Scan3R/3RScan"
    scan3r_img_projector = Scan3RIMGProjector(data_root_dir, split='train', use_rescan=True)
    step=5
    for idx in tqdm(range(len(scan3r_img_projector.scan_ids))):
        scan3r_img_projector.project(idx, step=5)