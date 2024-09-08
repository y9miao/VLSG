from curses import color_content
from itertools import count
from operator import is_
import os
import os.path as osp
from platform import node
import sys
from turtle import color
from matplotlib.pyplot import sca
from collections import Counter
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
    def __init__(self, cfg, split, retrieval_result_dir, temporal = False):
        self.cfg = cfg
        self.split = split
        
        # undefined patch anno id
        self.undefined = 0
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = ''
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' 
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
        self.temporal = temporal
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
        self.vis_out_dir = osp.join(self.retrieval_result_dir, 'vis_{}'.format(self.split))
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
    def visualize_patch_match(self, our_dir, scan_id,  frame_idx):
        color_right = [0, 255, 0]
        color_wrong = [255, 0, 0]
        color_no_ann = [0, 0, 0]
        
        # get retrieval result
        retrieval_record = self.retrieval_result[scan_id]['frames_retrieval'][frame_idx]
        temporal_scan_id = scan_id if not self.temporal else retrieval_record['temporal_scan_id']
        # get image
        image_path = self.image_paths[scan_id][frame_idx]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.img_rotate:
            image = image.transpose(1, 0, 2)
            image = np.flip(image, 1)
            
        img_w = image.shape[1]
        img_h = image.shape[0]
        patch_w = self.patch_w
        patch_h = self.patch_h
        
        pred_ids = retrieval_record['matched_obj_ids'] if not self.temporal \
            else retrieval_record['matched_obj_ids_temp']
        gt_anno = retrieval_record['gt_anno']
        
        gt_anno = gt_anno.reshape((patch_h, patch_w))
        pred_ids = pred_ids.reshape((patch_h, patch_w))
        pred_correct = pred_ids == gt_anno
        # pred_color_map
        pred_color = []
        pred_correct_color = []
        gt_color = []
        for i in range(patch_h):
            for j in range(patch_w):
                pred_id = pred_ids[i, j]
                gt_id = gt_anno[i, j]
                pred_color.append(self.obj_color[temporal_scan_id][pred_id])
                gt_color.append(self.obj_color[scan_id][gt_id] if gt_id != self.undefined else color_no_ann)
                if gt_id == self.undefined:
                    pred_correct_color.append(color_no_ann)
                else:
                    pred_correct_color.append(color_right if pred_correct[i, j] else color_wrong)
        
        pred_color = np.array(pred_color).reshape((patch_h, patch_w, 3)).astype(np.uint8)
        pred_correct_color = np.array(pred_correct_color).reshape((patch_h, patch_w, 3)).astype(np.uint8)
        gt_color = np.array(gt_color).reshape((patch_h, patch_w, 3)).astype(np.uint8)
        
        # resize color to the same size as the image
        pred_color = cv2.resize(pred_color, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        pred_correct_color = cv2.resize(pred_correct_color, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        gt_color = cv2.resize(gt_color, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # save image
        common.ensure_dir(our_dir)
        image_out_path = osp.join(our_dir, 'image.png')
        cv2.imwrite(image_out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        pred_color_out_path = osp.join(our_dir, 'pred_color.png')
        cv2.imwrite(pred_color_out_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
        pred_correct_color_out_path = osp.join(our_dir, 'pred_correct_color.png')
        cv2.imwrite(pred_correct_color_out_path, cv2.cvtColor(pred_correct_color, cv2.COLOR_RGB2BGR))
        gt_color_out_path = osp.join(our_dir, 'gt_color.png')
        cv2.imwrite(gt_color_out_path, cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
        
    # generate visualization file of room retrieval result of certain frame in certain scan
    def visualize_room_retrieval(self, scan_id, frame_idx, out_dir, 
                                 top_k = 5,):
        ## record the room retrieval result
        retrieval_record = self.retrieval_result[scan_id]['frames_retrieval'][frame_idx]
        room_scores = retrieval_record['room_score_scans_T'] if self.temporal \
            else retrieval_record['room_score_scans_NT']
        room_sorted_by_scores =  [item[0] for item in sorted(room_scores.items(), key=lambda x: x[1], reverse=True)]
        
        ## record scene_id and their scores
        out_lines = []
        for i, room_id in enumerate(room_sorted_by_scores):
            out_lines.append('{} with score: {}'.format(room_id, room_scores[room_id]))
        common.ensure_dir(out_dir)
        common.write_to_txt(osp.join(out_dir, 'room_retrieval_result.txt'), out_lines)
        
    def visualize_BestWorstK_Results(self, K = 5, th_counts = 4, th_objs = 2):
        def metric_score_successratio(data_item, scan_id):
            room_scores = data_item['room_score_scans_T'] if self.temporal \
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
            gt_anno = data_item['gt_anno']
            matched_obj_ids = data_item['matched_obj_ids'] if not self.temporal \
                else data_item['matched_obj_ids_temp']
            gt_anno_valid = gt_anno[gt_anno != self.undefined]
            matched_obj_ids_valid = matched_obj_ids[gt_anno != self.undefined]
            pred_obj_ids_unique, counts = np.unique(matched_obj_ids_valid, return_counts=True)
            num_correct = 0
            for i, obj_id in enumerate(pred_obj_ids_unique):
                counts_obj = counts[i]
                if counts_obj < th_counts:
                    continue
                is_correct = np.sum(matched_obj_ids_valid[matched_obj_ids_valid == obj_id] == \
                                    gt_anno_valid[matched_obj_ids_valid == obj_id]) > 0.3 * counts_obj
                num_correct += int(is_correct)
            ## match success ratio
            match_success_ratio = np.sum(matched_obj_ids_valid == gt_anno_valid) * 1.0 / gt_anno_valid.shape[0]
            score2 = num_correct*match_success_ratio*score_ratio if score1 > 0 else score_ratio
            ## target 
            return score1 +  score2
        def metric_score_entropy_successratio(data_item, scan_id):
            room_scores = data_item['room_score_scans_T'] if self.temporal \
                else data_item['room_score_scans_NT']
            room_sorted_by_scores =  [item[0] for item in \
                sorted(room_scores.items(), key=lambda x: x[1], reverse=True)]
            second_sim_scan = room_sorted_by_scores[1]
            target_scan_id = scan_id if not self.temporal else data_item['temporal_scan_id']
            # ## use ratio of the target scan's score to the second similar scan's score as the score1
            # if second_sim_scan not in room_scores or abs(room_scores[second_sim_scan]) < 1e-3:
            #     return None
            score_ratio = room_scores[target_scan_id] * 1.0 / room_scores[second_sim_scan]
            
            ## first make sure the room retrieval is valid
            score1 = 1000 * int(target_scan_id == room_sorted_by_scores[0])
            
            ## use num of correctly matched objects as the score2
            gt_anno = data_item['gt_anno']
            matched_obj_ids = data_item['matched_obj_ids'] if not self.temporal \
                else data_item['matched_obj_ids_temp']
            gt_anno_valid = gt_anno[gt_anno != self.undefined]
            matched_obj_ids_valid = matched_obj_ids[gt_anno != self.undefined]
            pred_obj_ids_unique, counts = np.unique(matched_obj_ids_valid, return_counts=True)
            num_correct = 0
            for i, obj_id in enumerate(pred_obj_ids_unique):
                counts_obj = counts[i]
                if counts_obj < th_counts:
                    continue
                is_correct = np.sum(matched_obj_ids_valid[matched_obj_ids_valid == obj_id] == \
                                    gt_anno_valid[matched_obj_ids_valid == obj_id]) > 0.3 * counts_obj
                num_correct += int(is_correct)
            ## match success ratio
            match_success_ratio = np.sum(matched_obj_ids_valid == gt_anno_valid) * 1.0 / gt_anno_valid.shape[0]
            score2 = num_correct*(match_success_ratio**2) *score_ratio if score1 > 0 else score_ratio
            
            ## entropy as score 3
            counts = Counter(gt_anno_valid)
            total = sum(counts.values())
            entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
            score3 = entropy
            ## target 
            return score1 +  score2 * score3
        
        def is_valid(data_item):
            gt_anno = data_item['gt_anno']
            matched_obj_ids = data_item['matched_obj_ids'] if not self.temporal \
                else data_item['matched_obj_ids_temp']
            # if there is no enough annotated/ predicted objects, then return False
            obj_ids_gt, counts_gt = np.unique(gt_anno, return_counts=True)
            obj_ids_pred, counts_pred = np.unique(matched_obj_ids, return_counts=True)
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
                score = metric_score_entropy_successratio(retrieval_record, scan_id)
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
            self.visualize_room_retrieval(scan_id, frame_idx, retrieval_out_dir, top_k=5)
            ## visualize patch match
            match_out_dir = osp.join(out_top_K_dir, scan_id, frame_idx, 'patch_match')

            self.visualize_patch_match(match_out_dir, scan_id, frame_idx)
        for data_item in worst_K_data_items:
            scan_id, frame_idx = data_item[0]
            ## visualize room retrieval
            retrieval_out_dir = osp.join(out_worst_K_dir, scan_id, frame_idx, 'room_retrieval')
            self.visualize_room_retrieval(scan_id, frame_idx, retrieval_out_dir, top_k=5)
            ## visualize patch match
            match_out_dir = osp.join(out_worst_K_dir, scan_id, frame_idx, 'patch_match')
            self.visualize_patch_match(match_out_dir, scan_id, frame_idx)
        
def main():
    os.environ["VLSG_SPACE"] = "/home/yang/big_ssd/Scan3R/VLSG"
    os.environ["Data_ROOT_DIR"] = "/home/yang/big_ssd/Scan3R/3RScan"
    retrieval_result_dir = "/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PaperMetric/OursStatistics/SGMatch_PAGRI_Top10_S1_E5_X10"
    cfg_file = "/home/yang/big_ssd/Scan3R/3RScan/out_room_retrieval/PaperMetric/OursStatistics/SGMatch_PAGRI_Top10_S1_E5_X10/inference_room_PARGI_statistics.yaml"
    
    from configs import update_config_room_retrival, config
    cfg = update_config_room_retrival(config, cfg_file, ensure_dir=False)
    visualizer = RoomRetrievalVisualizer(cfg, 'val', retrieval_result_dir)
    
    # scan_id = '0cac7584-8d6f-2d13-8df8-c05e4307b418'
    # frame_idx = '000000'
    # patch_match_out_dir = osp.join(visualizer.vis_out_dir, 'patch_match', scan_id, frame_idx)
    # visualizer.visualize_patch_match(patch_match_out_dir, scan_id, frame_idx)
    # room_retrie_out_dir = osp.join(visualizer.vis_out_dir, 'room_retrieval', scan_id, frame_idx)
    # visualizer.visualize_room_retrieval(scan_id, frame_idx, room_retrie_out_dir)
    
    visualizer.visualize_BestWorstK_Results(K = 10)
    
if __name__ == "__main__":
    main()
        