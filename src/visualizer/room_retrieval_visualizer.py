from curses import color_content
from itertools import count
from operator import is_
import os
import os.path as osp
from platform import node
import sys
from turtle import color
from matplotlib.pyplot import sca

import numpy as np
from pyparsing import line
import pyviz3d.visualizer as viz
import open3d as o3d
import cv2
import copy

from sympy import N

sys.path.append('..')
sys.path.append('../..')
from utils import common, scan3r
from configs import update_config_room_retrival, config

class RoomRetrievalVisualizer():
    def __init__(self, cfg, split, retrieval_result_dir):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
        # sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        self.sgaligner_modules = cfg.sgaligner.modules
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
        
        # scene_dir
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        # 2D images
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        # 2D_patch
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        # 2D patch anno
        if self.img_rotate:
            self.patch_h = self.image_patch_w
            self.patch_w = self.image_patch_h
            # self.image_w = self.cfg.data.img.h
            # self.image_h = self.cfg.data.img.w
        else:
            self.patch_h = self.image_patch_h
            self.patch_w = self.image_patch_w
        self.patch_num = self.patch_h * self.patch_w
        
        # cross scenes cfg
        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes
        self.num_negative_samples = cfg.data.cross_scene.num_negative_samples
        
        # if split is val, then use all object from other scenes as negative samples
        # if room_retrieval, then use load additional data items
        self.room_retrieval = False
        if split == 'val' or split == 'test':
            self.num_negative_samples = -1
            self.room_retrieval = True
            self.room_retrieval_epsilon_th = cfg.val.room_retrieval.epsilon_th
        
        # scans info
        self.temporal = cfg.data.temporal
        self.rescan = cfg.data.rescan
        scan_info_file = osp.join(self.scans_files_dir, '3RScan.json')
        all_scan_data = common.load_json(scan_info_file)
        self.refscans2scans = {}
        self.scans2refscans = {}
        self.all_scans_split = []
        for scan_data in all_scan_data:
            ref_scan_id = scan_data['reference']
            self.refscans2scans[ref_scan_id] = [ref_scan_id]
            self.scans2refscans[ref_scan_id] = ref_scan_id
            for scan in scan_data['scans']:
                self.refscans2scans[ref_scan_id].append(scan['reference'])
                self.scans2refscans[scan['reference']] = ref_scan_id
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        self.all_scans_split = []
        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            self.all_scans_split += self.refscans2scans[ref_scan]
        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split
            
        # load 2D image paths
        self.image_paths = {}
        for scan_id in self.all_scans_split:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id)
        # 2D image pose
        self.img_poses = {}
        for scan_id in self.all_scans_split:
            frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id)
            self.img_poses[scan_id] = {}
            for frame_idx in frame_idxs:
                self.img_poses[scan_id][frame_idx] = scan3r.load_pose(self.scans_scenes_dir, scan_id, frame_idx)
        
        # load transform matrix from rescan to ref
        self.trans_rescan2ref = scan3r.read_transform_mat(scan_info_file) 
        
        # load 3D anno paths and scene graph paths
        self.pcs_anno_paths = {}
        self.sg_paths = {}
        for scan_id in self.all_scans_split:
            # load 3D anno paths
            self.pcs_anno_paths[scan_id] = osp.join(self.scans_scenes_dir, scan_id, 'data.npy')
            # load 3D scene graph
            self.sg_paths[scan_id] = osp.join(self.scans_files_dir_mode, 'data/{}.pkl'.format(scan_id))
            
        # load object colors
        def Hex_to_RGB(hex):
            r = int(hex[1:3],16)
            g = int(hex[3:5],16)
            b = int(hex[5:7], 16)
            return [r, g, b]
        obj_json_filename = 'objects.json'
        obj_json = common.load_json(osp.join(self.scans_files_dir, obj_json_filename))['scans']
        self.obj_color = {}
        for scan_info in obj_json:
            scan_id = scan_info['scan']
            obj_colors = {}
            for obj_info in scan_info['objects']:
                obj_id = int(obj_info['id'])
                obj_color = np.array(Hex_to_RGB(obj_info['ply_color']))
                obj_colors[obj_id] = obj_color
            self.obj_color[scan_id] = obj_colors
            
        # retrieval_result_dir
        self.retrieval_result_dir = retrieval_result_dir
        self.retrieval_result_file = osp.join(self.retrieval_result_dir, 
                                              'retrieval_record_{}.pkl'.format(self.split))
        self.retrieval_result = common.load_pkl_data(self.retrieval_result_file)
        # vis out dir
        self.vis_out_dir = osp.join(self.retrieval_result_dir, 'vis')
        common.ensure_dir(self.vis_out_dir)

    def get_scene_info(self, scan_id, th_z = 2.0):
        # load 3D anno
        pcs_anno = np.load(self.pcs_anno_paths[scan_id])
        ## filter out unannotated points
        obj_ids = pcs_anno['objectId']
        pcs_anno = pcs_anno[obj_ids != self.undefined]
        
        # load 3D scene graph
        scene_graph = common.load_pkl_data(self.sg_paths[scan_id])
        
        # get scene pcs 
        scene_points = np.stack([pcs_anno['x'], pcs_anno['y'], pcs_anno['z']]).transpose((1, 0))
        if scan_id in self.trans_rescan2ref:
            scene_points_aug = np.concatenate([scene_points, np.ones((scene_points.shape[0], 1))], axis=1)
            T_ref2rescan = np.linalg.inv(self.trans_rescan2ref[scan_id]).astype(np.float32)
            scene_points[:,:3] = np.asarray(scene_points_aug@ T_ref2rescan).astype(np.float32)[:,:3]
        scene_colors = np.stack([pcs_anno['red'], pcs_anno['green'], pcs_anno['blue']]).transpose((1, 0))
        ## remove ceil for better visualization
        z_min = np.min(scene_points[:, 2])
        z_valid = scene_points[:, 2] < (z_min + th_z)
        scene_points = scene_points[z_valid]
        scene_colors = scene_colors[z_valid]
        pcs_anno = pcs_anno[z_valid]
        return pcs_anno, scene_graph, scene_points, scene_colors
    
    def generate_camera_frustum(self, cam_extrinsics, target_scene_center,
                                cam_color = [0, 0, 0],
                                img_width_m = 0.7, cam_depth_range = 0.8, 
                                point_size = 0.05):
        img_height_m = img_width_m * self.image_h / self.image_w
        
        # get points in camera frustum
        image_center = np.array([0, 0, 0])
        image_corners =  np.array([
            [img_width_m/2.0, img_height_m/2.0, cam_depth_range],
            [-img_width_m/2.0, img_height_m/2.0, cam_depth_range],
            [-img_width_m/2.0, -img_height_m/2.0, cam_depth_range],
            [img_width_m/2.0, -img_height_m/2.0, cam_depth_range],
        ])
        # transform to world coord
        image_center =  image_center @ (cam_extrinsics[:3, :3]).T  + cam_extrinsics[:3, 3].T
        image_corners = image_corners @ (cam_extrinsics[:3, :3]).T  + cam_extrinsics[:3, 3].T
        image_center = image_center + target_scene_center.reshape(1, 3)
        image_corners = image_corners + target_scene_center.reshape(1, 3)
        # get lines in camera frustum
        line_ps_start = []
        line_ps_end = []
        for i in range(4):
            line_ps_start.append(image_center.reshape(3))
            line_ps_end.append(image_corners[i].reshape(3))
        for i in range(4):
            line_ps_start.append(image_corners[i].reshape(3))
            line_ps_end.append(image_corners[(i+1)%4].reshape(3))
        line_ps_start = np.array(line_ps_start)
        line_ps_end = np.array(line_ps_end)
        # generate point balls
        points_center, colors_center = self.generate_ball_pcl(image_center, point_size, cam_color)
        # for i in range(image_corners.shape[0]):
        #     points_corner, colors_corner = self.generate_ball_pcl(image_corners[i], point_size, cam_color)
        #     points_center = np.concatenate([points_center, points_corner], axis=0)
        #     colors_center = np.concatenate([colors_center, colors_corner], axis=0)
        return line_ps_start, line_ps_end, points_center, colors_center
            
    def generate_ball_pcl(self, position, radius, color, resolution=5):
        # generate ball
        points = []
        ## generate ball points
        for i in range(0, 360, resolution):
            for j in range(0, 360, resolution):
                x = radius * np.cos(np.deg2rad(i)) * np.sin(np.deg2rad(j))
                y = radius * np.sin(np.deg2rad(i)) * np.sin(np.deg2rad(j))
                z = radius * np.cos(np.deg2rad(j))
                points.append([x, y, z])
        points = np.array(points)
        ## translate ball
        points += position
        colors = np.array([color for _ in range((points.shape[0]))])
        return points, colors
    
    def generate_scene_graph_pcs(self, scan_id, position, pcs_anno, scene_points, pcs_num_th = 100, ball_size = 0.2):
        nodes_pcs = None
        nodes_colors = None
        
        obj_ids = pcs_anno['objectId']
        for obj_id in self.obj_color[scan_id]:
            obj_points = scene_points[obj_ids == obj_id]
            if obj_points.shape[0] < pcs_num_th:
                continue
            obj_points += position
            obj_color = self.obj_color[scan_id][obj_id]
            obj_position = np.mean(obj_points, axis=0)
            node_pcs, node_colors = self.generate_ball_pcl(obj_position, ball_size, obj_color)
            
            nodes_pcs = node_pcs if nodes_pcs is None else np.concatenate([nodes_pcs, node_pcs], axis=0)
            nodes_colors = node_colors if nodes_colors is None else np.concatenate([nodes_colors, node_colors], axis=0)
        return nodes_pcs, nodes_colors
    
    def generate_image_pcs(self, image, position, resolution = 1, dist=0.001):
        # image: to be transformed to pcs
        # position: position of the image left up corner, H-Z, W-X, D-Y alignment
        # resolution: resolution of the pixels
        # dist: distance between two pixels
        img_w = image.shape[1]
        img_h = image.shape[0]
        points = []
        colors = []
        for i in range(0, img_h, resolution):
            for j in range(0, img_w, resolution):
                x = (j) * dist # along x-axis, same direction as W
                y = (- i) * dist # along y-axis, opposite direction as H
                z = 0
                points.append([x, y, z])
                colors.append(image[i, j])
        points = np.array(points)
        points += position
        colors = np.array(colors)
        return points, colors
    
    def generate_image_pred_pcs(self, image, scan_id, 
                                position_pred, position_correct,
                                pred_obj_id_flatten, gt_obj_id_flatten, 
                                resolution = 1, dist=0.001):
        points_pred = []
        colors_pred = []
        points_correct = []
        colors_correct = []
        color_right = [0, 255, 0]
        color_wrong = [255, 0, 0]
        color_no_ann = [0, 0, 0]
        
        img_w = image.shape[1]
        img_h = image.shape[0]
        patch_w = self.patch_w
        patch_h = self.patch_h
        patch_w_size = img_w * 1.0 / patch_w
        patch_h_size = img_h * 1.0 / patch_h
        pred = pred_obj_id_flatten.reshape((patch_h, patch_w))
        gt_anno = gt_obj_id_flatten.reshape((patch_h, patch_w))
        for i in range(0, img_h, resolution):
            for j in range(0, img_w, resolution):
                patch_h_i = min(int(i / patch_h_size), patch_h - 1) 
                patch_w_j = min(int(j / patch_w_size), patch_w - 1)
                is_correct = pred[patch_h_i, patch_w_j] == gt_anno[patch_h_i, patch_w_j]
                
                pred_x = (j) * dist # along x-axis, same direction as W
                pred_y = (- i) * dist # along y-axis, opposite direction as H
                pred_z = 0
                points_pred.append([pred_x, pred_y, pred_z])
                pred_obj_color = self.obj_color[scan_id][pred[patch_h_i, patch_w_j]]
                colors_pred.append(pred_obj_color)
                
                correct_x = (j) * dist # along x-axis, same direction as W
                correct_y = (- i) * dist # along y-axis, opposite direction as H
                correct_z = 0
                points_correct.append([correct_x, correct_y, correct_z])
                if gt_anno[patch_h_i, patch_w_j] == self.undefined:
                    colors_correct.append(color_no_ann)
                else:
                    if is_correct:
                        colors_correct.append(color_right)
                    else:
                        colors_correct.append(color_wrong)
        points_pred = np.array(points_pred)
        points_pred += position_pred
        colors_pred = np.array(colors_pred)
        points_correct = np.array(points_correct)
        points_correct += position_correct
        colors_correct = np.array(colors_correct)
        return points_pred, colors_pred, points_correct, colors_correct
    
    def generate_patch_match_arrows(self, scan_id, pcs_anno, scene_points, pred_obj_ids_flatten, gt_obj_id_flatten,
                                    image_position, image_3Dsize_w, image_3Dsize_h, obj_ball_size, th_counts = 2):
        def map_patch_pos_to_3D_pos(patch_pos, image_position, image_size_w, image_size_h):
            x = (patch_pos[1]) * image_size_w / self.patch_w
            y = -(patch_pos[0]) * image_size_h / self.patch_h
            z = 0
            return np.array([x, y, z]) + image_position
        obj3D_ids_anno = pcs_anno['objectId']
        
        # get patch match result
        color_right = np.array([0, 255, 0])
        color_wrong = np.array([255, 0, 0])
        arrows = []
        obj_patch_spheres_points = None
        obj_patch_spheres_colors = None
        pred_obj_ids = pred_obj_ids_flatten.reshape((self.patch_h, self.patch_w))
        gt_obj_ids = gt_obj_id_flatten.reshape((self.patch_h, self.patch_w))
        pred_obj_ids_unique, counts = np.unique(pred_obj_ids_flatten[gt_obj_id_flatten!=self.undefined], return_counts=True)
        for obj_idx in range(pred_obj_ids_unique.shape[0]):
            obj_id = pred_obj_ids_unique[obj_idx]
            count = counts[obj_idx]
            idxs_3D = (obj3D_ids_anno == obj_id)
            if obj_id == self.undefined or count<=th_counts or np.sum(idxs_3D) < 100:
                continue
            obj_positions = np.argwhere(pred_obj_ids == obj_id)
            obj_pos_mean = np.mean(obj_positions, axis=0)
            # find nearst obj_positions of the obj to obj_pos_mean
            obj_positions_relative = obj_positions - obj_pos_mean
            dist_norm = np.linalg.norm(obj_positions_relative, axis=1)
            nearest_idx = np.argsort(dist_norm)[0]
            obj_patch_pos = obj_positions[nearest_idx]
            obj_patch_pos_3D = map_patch_pos_to_3D_pos(obj_patch_pos, image_position, 
                                image_3Dsize_w, image_3Dsize_h)
            # is prediction correct
            is_correct = np.sum( pred_obj_ids[pred_obj_ids == obj_id] == 
                    gt_obj_ids[pred_obj_ids == obj_id]) > 0.3 * count
            if is_correct:
                # get positions of the obj in 3D
                obj_3D_points = scene_points[idxs_3D]
                obj_3D_center = np.mean(obj_3D_points, axis=0)
                ## make the arrow shorter so that the head of arrow is outside the object ball
                arrow_start = obj_patch_pos_3D
                arrow_end_in_ball = obj_3D_center
                arrow_length = np.linalg.norm(arrow_end_in_ball - arrow_start)
                lambda_interpo = (arrow_length - obj_ball_size) / arrow_length
                arrow_end = arrow_start + lambda_interpo * (arrow_end_in_ball - arrow_start)
                
                ## also generate the ball at the obj patch
                obj_balls_at_patch, obj_colors_at_patch = self.generate_ball_pcl(arrow_start, obj_ball_size,
                                                                                self.obj_color[scan_id][obj_id])
                obj_patch_spheres_points = obj_balls_at_patch if obj_patch_spheres_points is None \
                    else np.concatenate([obj_patch_spheres_points, obj_balls_at_patch], axis=0)
                obj_patch_spheres_colors = obj_colors_at_patch if obj_patch_spheres_colors is None \
                    else np.concatenate([obj_patch_spheres_colors, obj_colors_at_patch], axis=0)
                # get arrow
                arrow = [arrow_start, arrow_end, color_right if is_correct else color_wrong]
                arrows.append(arrow)
                
        return arrows, obj_patch_spheres_points, obj_patch_spheres_colors
            
            
    # generate visualization file of patch match result of certain frame in certain scan
    def visualize_patch_match(self, our_dir, scan_id, frame_idx):
        
        # get retrieval result
        retrieval_record = self.retrieval_result[scan_id]['frames_retrieval'][frame_idx]
        # get image
        image_path = self.image_paths[scan_id][frame_idx]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.img_rotate:
            image = image.transpose(1, 0, 2)
            image = np.flip(image, 1)
            
        # load 3D anno
        pcs_anno, scene_graph, scene_points, scene_colors = self.get_scene_info(scan_id, th_z = 2.5)
        ## get scene position
        position_pcs = np.array([0, 0, 0])
        scene_center = np.mean(scene_points, axis=0)
        scene_size = np.max(scene_points, axis=0) - np.min(scene_points, axis=0)
        scene_max = np.max(scene_points, axis=0)
        scene_min = np.min(scene_points, axis=0)
        
        # get scene graph pcs
        # position_sg = np.array([scene_center[0] + scene_size[0] * 1.1, scene_center[1], scene_center[2]])
        position_sg = position_pcs
        node_radius = 0.2
        sg_pcs, sg_colors = self.generate_scene_graph_pcs(
                scan_id, position_sg, pcs_anno, scene_points, ball_size=node_radius, pcs_num_th = 100)
        ## get scene graph position
        sg_center = np.mean(sg_pcs, axis=0)
        sg_size = np.max(sg_pcs, axis=0) - np.min(sg_pcs, axis=0)
        sg_max = np.max(sg_pcs, axis=0)
        sg_min = np.min(sg_pcs, axis=0)
    
        # generate image pcs
        ## get image positions 
        adaptive_dist = scene_size[1] / (2 * image.shape[0])
        img_pcs_size_w = adaptive_dist * image.shape[1]
        img_pcs_size_h = adaptive_dist * image.shape[0]
        ## set image position near the scene center in X-Y plane
        image_position = np.array([scene_center[0] - img_pcs_size_w * 1.6, # X align with center of the scene
                                   scene_min[1] - 0.3, # Y align with the min of the scene
                                   scene_min[2] - 0.2]) # Z align with the center of the scene
        ## make the image half large as the scene, dist*img_h = scene_size[1]/2 in Y direction
        image_points, image_colors = self.generate_image_pcs(image, image_position, resolution=1, dist=adaptive_dist)
        
        # generate anno and pred image pcs
        ## get image positions 
        position_pred = np.array([scene_center[0] - img_pcs_size_w/2.0, # X align with center of the scene graph
                                   image_position[1], # Y align with the min of the scene graph
                                   image_position[2]]) # Z align with the center of the scene graph
        position_correct = np.array([position_pred[0] + 1.1 * img_pcs_size_w,
                                     position_pred[1],
                                    position_pred[2]])
        ## get anno and pred
        gt_obj_id_flatten = retrieval_record['gt_anno'].cpu().numpy()
        pred_obj_id_flatten = retrieval_record['matched_obj_obj_ids']
        points_pred, colors_pred, points_correct, colors_correct = self.generate_image_pred_pcs(
            image, scan_id, position_pred, position_correct,
            pred_obj_id_flatten, gt_obj_id_flatten, dist=adaptive_dist)
        
        ## get arrows
        arrows,obj_patch_spheres_points, obj_patch_spheres_colors = \
            self.generate_patch_match_arrows(scan_id, pcs_anno, scene_points, 
                    pred_obj_id_flatten, gt_obj_id_flatten, image_position, 
                    img_pcs_size_w, img_pcs_size_h, 0.1)
        
        
        # add camera frustum
        cam_extrinsics = self.img_poses[scan_id][frame_idx]
        target_scene_origin = position_pcs
        line_ps_start, line_ps_end, points_frustum, colors_frustum = \
            self.generate_camera_frustum(cam_extrinsics, target_scene_origin)
        
        vis = viz.Visualizer()
        vis.add_points(
            'scene', 
            positions=scene_points,
            colors=scene_colors)
        vis.add_points(
            'scene_graph', 
            positions=sg_pcs,
            colors=sg_colors)
        vis.add_points(
            'image', 
            positions=image_points,
            colors=image_colors)
        vis.add_points(
            'pred_obj', 
            positions=points_pred,
            colors=colors_pred)
        vis.add_points(
            'correct_obj', 
            positions=points_correct,
            colors=colors_correct)
        # add image frustum
        vis.add_points(
            'frustum', 
            positions=points_frustum,
            colors=colors_frustum)
        vis.add_lines(
            'frustum_lines', 
            line_ps_start, line_ps_end)
        # add arrows
        for i, arrow in enumerate(arrows):
            vis.add_arrow(
                'arrow_{}'.format(i), 
                arrow[0], arrow[1], arrow[2],
                head_width=0.1,
                stroke_width=0.05)
        if obj_patch_spheres_points is not None:
            vis.add_points(
                'obj_patch_spheres', 
                positions=obj_patch_spheres_points,
                colors=obj_patch_spheres_colors)
        
        
        # save visualization file
        # vis_folder = osp.join(self.vis_out_dir, scan_id, str(frame_idx), 'match')
        common.ensure_dir(our_dir)
        vis.save(our_dir)
        
    # generate visualization file of room retrieval result of certain frame in certain scan
    def visualize_room_retrieval(self, scan_id, frame_idx, out_dir, 
                                 top_k = 5, temporal = False):
        # scan_id: scan_id of the query frame
        # top_k: show top k retrieved rooms
        # temporal: if True, show temporal retrieval result
        
        # get retrieval result
        retrieval_record = self.retrieval_result[scan_id]['frames_retrieval'][frame_idx]
        # get image
        image_path = self.image_paths[scan_id][frame_idx]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.img_rotate:
            image = image.transpose(1, 0, 2)
            image = np.flip(image, 1)
            
        # load 3D anno of the scene
        pcs_anno, scene_graph, scene_points, scene_colors = self.get_scene_info(scan_id)
        
        # get room score
        room_scores = retrieval_record['room_score_scans_T'] if temporal \
            else retrieval_record['room_score_scans_NT']
        room_sorted_by_scores_NT =  [item[0] for item in sorted(room_scores.items(), key=lambda x: x[1], reverse=True)]
        top_k_scan_ids = room_sorted_by_scores_NT[:top_k]
        ## if target scan_id is not in the top k, add it to the top k
        if scan_id not in top_k_scan_ids:
             top_k_scan_ids.append(scan_id)
        
        # get top k scene pcs
        top_k_scene_pcs = {}
        for candidate_scan_id in top_k_scan_ids:
            pcs_anno, scene_graph, scene_points, scene_colors = self.get_scene_info(candidate_scan_id)
            top_k_scene_pcs[candidate_scan_id] = {
                'pcs_anno': pcs_anno, 'scene_graph': scene_graph,
                'scene_points': scene_points, 'scene_colors': scene_colors}
            
        top_k_scene_pcs_out = {}
        # set positions for the top k scene pcs
        last_scene_origin = np.array([0, 0, 0])
        last_scene_max = np.array([0, 0, 0])
        last_scene_min = np.array([0, 0, 0])
        for i, candidate_scan_id in enumerate(top_k_scan_ids):
            pcs_anno, scene_graph, scene_points, scene_colors = (top_k_scene_pcs[candidate_scan_id]['pcs_anno'],
                                                                top_k_scene_pcs[candidate_scan_id]['scene_graph'],
                                                                top_k_scene_pcs[candidate_scan_id]['scene_points'],
                                                                top_k_scene_pcs[candidate_scan_id]['scene_colors'])
            # get scene position at origin coord of the scene
            model_center = np.mean(scene_points, axis=0)
            model_size = np.max(scene_points, axis=0) - np.min(scene_points, axis=0)
            model_max = np.max(scene_points, axis=0)
            model_min = np.min(scene_points, axis=0)
            # set position in vis coord
            position = np.array([
                last_scene_max[0] - model_min[0] + 0.4, 
                last_scene_min[1] - model_min[1],  # Y and Z align with the last scene
                last_scene_min[2] - model_min[2]])
            scene_points_out = scene_points + position
            
            # update last_scene_origin, last_scene_size, last_scene_max
            last_scene_origin = position
            last_scene_max = model_max + position
            last_scene_min = model_min + position
            
            top_k_scene_pcs_out[candidate_scan_id] = {
                'scene_points': scene_points_out, 
                'scene_colors': scene_colors,
                'model_size': model_size,
                'scene_origin': last_scene_origin,
                'scene_max': last_scene_max,
                'scene_min': last_scene_min}

            # get labels
            text_content = 'scan: {}, score: {:.3f}'.format(candidate_scan_id[:8], 
                                                      room_scores[candidate_scan_id]*1.0/self.patch_num)
            text_position = np.array([
                last_scene_min[0] + 0.5 * (last_scene_origin[0] - last_scene_min[0]),
                last_scene_min[1] - 0.1,
                last_scene_min[2]
            ])
            top_k_scene_pcs_out[candidate_scan_id]['text'] = {
                'content': text_content, 'position': text_position}
        
        # add image pcs
        ## get image positions 
        target_scene_pcs_out = top_k_scene_pcs_out[scan_id]
        target_model_size = target_scene_pcs_out['model_size']
        target_scene_min = target_scene_pcs_out['scene_min']
        target_scene_origin = target_scene_pcs_out['scene_origin']
        adaptive_dist = target_model_size[1] / (image.shape[0])
        ## set image position below the target scene in Y aixs, X-Y plane   
        image_position = np.array([target_scene_min[0] + 0.5 * (target_scene_origin[0] - target_scene_min[0]),
                                   target_scene_min[1] - 0.3, # Y align with the min of the scene
                                   target_scene_origin[2]]) # Z align with the center of the scene
        image_points, image_colors = self.generate_image_pcs(
            image, image_position, resolution=1, dist=adaptive_dist)
        
        # add image frustum
        cam_extrinsics = self.img_poses[scan_id][frame_idx]
        line_ps_start, line_ps_end, points_frustum, colors_frustum = \
            self.generate_camera_frustum(cam_extrinsics, target_scene_origin)
            
        vis = viz.Visualizer()
        # add scene pcs
        for candidate_scan_id in top_k_scan_ids:
            scene_points = top_k_scene_pcs_out[candidate_scan_id]['scene_points']
            scene_colors = top_k_scene_pcs_out[candidate_scan_id]['scene_colors']
            vis.add_points(
                name = 'pc' + candidate_scan_id[:8], 
                positions=scene_points,
                colors=scene_colors)
            label_content = top_k_scene_pcs_out[candidate_scan_id]['text']['content']
            label_position = top_k_scene_pcs_out[candidate_scan_id]['text']['position']
            vis.add_labels( 
                name='la' + candidate_scan_id[:8], 
                labels= [label_content],
                positions=[label_position],
                colors=[np.array([0, 0, 0])])
        # add image pcs
        vis.add_points(
            'image', 
            positions=image_points,
            colors=image_colors)
        # add image frustum
        vis.add_points(
            'frustum', 
            positions=points_frustum,
            colors=colors_frustum)
        vis.add_lines(
            'frustum_lines', 
            line_ps_start, line_ps_end)
        
        common.ensure_dir(out_dir)
        vis.save(out_dir)
        
    def visualize_BestWorstK_Results(self, K = 5, temporal = False, th_counts = 4, th_objs = 2):
        def metric_score(data_item, scan_id):
            room_scores = data_item['room_score_scans_T'] if temporal \
                else data_item['room_score_scans_NT']
            room_sorted_by_scores =  [item[0] for item in \
                sorted(room_scores.items(), key=lambda x: x[1], reverse=True)]
            second_sim_scan = room_sorted_by_scores[1]
            target_scan_id = scan_id
            # ## use ratio of the target scan's score to the second similar scan's score as the score1
            # if second_sim_scan not in room_scores or abs(room_scores[second_sim_scan]) < 1e-3:
            #     return None
            score_ratio = room_scores[target_scan_id] * 1.0 / room_scores[second_sim_scan]
            
            ## first make sure the room retrieval is valid
            score1 = 1000 * int(target_scan_id == room_sorted_by_scores[0])
            
            ## use num of correctly matched objects as the score2
            gt_anno = data_item['gt_anno'].cpu().numpy()
            matched_obj_obj_ids = data_item['matched_obj_obj_ids']
            gt_anno_valid = gt_anno[gt_anno != self.undefined]
            matched_obj_obj_ids_valid = matched_obj_obj_ids[gt_anno != self.undefined]
            pred_obj_ids_unique, counts = np.unique(matched_obj_obj_ids_valid, return_counts=True)
            num_correct = 0
            for i, obj_id in enumerate(pred_obj_ids_unique):
                counts_obj = counts[i]
                if counts_obj < th_counts:
                    continue
                is_correct = np.sum(matched_obj_obj_ids_valid[matched_obj_obj_ids_valid == obj_id] == \
                                    gt_anno_valid[matched_obj_obj_ids_valid == obj_id]) > 0.3 * counts_obj
                num_correct += int(is_correct)
            ## match success ratio
            match_success_ratio = np.sum(matched_obj_obj_ids_valid == gt_anno_valid) * 1.0 / gt_anno_valid.shape[0]
            
            score2 = num_correct*match_success_ratio*score_ratio if score1 > 0 else score_ratio
            
            ## target 
            return score1 +  score2
        
        def is_valid(data_item):
            gt_anno = data_item['gt_anno'].cpu().numpy()
            matched_obj_obj_ids = data_item['matched_obj_obj_ids']
            # if there is no enough annotated/ predicted objects, then return False
            obj_ids_gt, counts_gt = np.unique(gt_anno, return_counts=True)
            obj_ids_pred, counts_pred = np.unique(matched_obj_obj_ids, return_counts=True)
            ## filter out undefined and counts < th_counts
            invalid_idxs_gt = np.logical_or(obj_ids_gt == self.undefined, counts_gt < th_counts)
            obj_ids_gt = obj_ids_gt[invalid_idxs_gt]
            invalid_idxs_pred = np.logical_or(obj_ids_pred == self.undefined, counts_pred < th_counts)
            obj_ids_pred = obj_ids_pred[invalid_idxs_pred]
            if obj_ids_gt.shape[0] < th_objs or obj_ids_pred.shape[0] < th_objs:
                return False
            return True
        # get scores of all data items
        data_item_scores_best = {}
        data_item_scores_worst = {}
        for scan_id in self.scan_ids:
            room_retrieval_record_scan = self.retrieval_result[scan_id]['frames_retrieval']
            metric_score_best = None
            metric_score_worst = None
            for frame_idx in room_retrieval_record_scan:
                if not is_valid(room_retrieval_record_scan[frame_idx]):
                    continue
                retrieval_record = room_retrieval_record_scan[frame_idx]
                score = metric_score(retrieval_record, scan_id)
                if score is not None:
                    if metric_score_best is None or score > metric_score_best[1]:
                        metric_score_best = [(scan_id, frame_idx), score]
                    if metric_score_worst is None or score < metric_score_worst[1]:
                        metric_score_worst = [(scan_id, frame_idx), score]
            if metric_score_best is not None:
                data_item_scores_best[metric_score_best[0]] = metric_score_best[1]
            if metric_score_worst is not None:
                data_item_scores_worst[metric_score_worst[0]] = metric_score_worst[1]
                
        # select top K and worst k data items
        ## select top K and worst k
        best_sorted = sorted(data_item_scores_best.items(), key=lambda x: x[1], reverse=True)
        top_K_data_items = best_sorted[:K]
        worst_sorted = sorted(data_item_scores_worst.items(), key=lambda x: x[1], reverse=False)
        worst_K_data_items = worst_sorted[:K]
        
        # visualize top K and worst K
        out_top_K_dir = osp.join(self.vis_out_dir, 'top_K')
        out_worst_K_dir = osp.join(self.vis_out_dir, 'worst_K')
        for data_item in top_K_data_items:
            scan_id, frame_idx = data_item[0]
            ## visualize room retrieval
            retrieval_out_dir = osp.join(out_top_K_dir, scan_id, frame_idx, 'room_retrieval')
            self.visualize_room_retrieval(scan_id, frame_idx, retrieval_out_dir, top_k=5, temporal=temporal)
            ## visualize patch match
            match_out_dir = osp.join(out_top_K_dir, scan_id, frame_idx, 'patch_match')
            self.visualize_patch_match(match_out_dir, scan_id, frame_idx)
        for data_item in worst_K_data_items:
            scan_id, frame_idx = data_item[0]
            ## visualize room retrieval
            retrieval_out_dir = osp.join(out_worst_K_dir, scan_id, frame_idx, 'room_retrieval')
            self.visualize_room_retrieval(scan_id, frame_idx, retrieval_out_dir, top_k=5, temporal=temporal)
            ## visualize patch match
            match_out_dir = osp.join(out_worst_K_dir, scan_id, frame_idx, 'patch_match')
            self.visualize_patch_match(match_out_dir, scan_id, frame_idx)
        
def main():
    os.environ["VLSG_SPACE"] = "/home/yang/big_ssd/Scan3R/VLSG"
    os.environ["Scan3R_ROOT_DIR"] = "/home/yang/big_ssd/Scan3R/3RScan"
    retrieval_result_dir = "/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PatchObjMatch/NOI_Aug3D_Step2_E17"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/test/Vis/room_retrie_cfg.yaml"
    
    from configs import update_config_room_retrival, config
    cfg = update_config_room_retrival(config, cfg_file, ensure_dir=False)
    visualizer = RoomRetrievalVisualizer(cfg, 'val', retrieval_result_dir)
    
    # scan_id = '0cac7584-8d6f-2d13-8df8-c05e4307b418'
    # frame_idx = '000000'
    # patch_match_out_dir = osp.join(visualizer.vis_out_dir, 'patch_match', scan_id, frame_idx)
    # visualizer.visualize_patch_match(patch_match_out_dir, scan_id, frame_idx)
    # room_retrie_out_dir = osp.join(visualizer.vis_out_dir, 'room_retrieval', scan_id, frame_idx)
    # visualizer.visualize_room_retrieval(scan_id, frame_idx, room_retrie_out_dir)
    
    visualizer.visualize_BestWorstK_Results()
    
if __name__ == "__main__":
    main()
        