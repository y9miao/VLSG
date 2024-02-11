# get into VLSG space for scan3r data info
import os
import os.path as osp
import sys
from tracemalloc import start
from sklearn.utils import resample
from yaml import scan
vlsg_dir = "/home/yang/big_ssd/Scan3R/VLSG"
sys.path.insert(0, vlsg_dir)
from utils import common, scan3r

import numpy as np
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms as tvf
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal, Tuple, List, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import cv2
# config
from configs import update_config, config

class Scan3rPatchDepthGenerator():
    def __init__(self, cfg, split, vis = False):
        self.cfg = cfg
        self.vis = vis
        
        # 3RScan data info
        ## sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        ## data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        ## scans info
        self.rescan = cfg.data.rescan
        scan_info_file = osp.join(self.scans_files_dir, '3RScan.json')
        all_scan_data = common.load_json(scan_info_file)
        self.refscans2scans = {}
        self.scans2refscans = {}
        for scan_data in all_scan_data:
            ref_scan_id = scan_data['reference']
            self.refscans2scans[ref_scan_id] = [ref_scan_id]
            self.scans2refscans[ref_scan_id] = ref_scan_id
            if self.rescan:
                for scan in scan_data['scans']:
                    self.refscans2scans[ref_scan_id].append(scan['reference'])
                    self.scans2refscans[scan['reference']] = ref_scan_id
        self.resplit = "resplit_" if cfg.data.resplit else ""
        if self.rescan:
            ref_scans = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
            self.scan_ids = []
            for ref_scan in ref_scans:
                self.scan_ids += self.refscans2scans[ref_scan]
        else:
            self.scan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id)
        
        # load patch anno
        self.patch_anno = {}
        self.patch_anno_folder = osp.join(self.scans_files_dir, 'patch_anno/patch_anno_16_9')
        for scan_id in self.scan_ids:
            self.patch_anno[scan_id] = common.load_pkl_data(osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id)))
        
        # load gt anno path
        self.gt_anno_path = {}
        self.gt_anno_folder = osp.join(self.scans_files_dir, 'gt_projection/obj_id_pkl')
        for scan_id in self.scan_ids:
            self.gt_anno_path[scan_id] = osp.join(self.gt_anno_folder, "{}.pkl".format(scan_id))
        self.undefined_obj_id = 0
        
        # model info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
        
        # image preprocessing 
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.step = self.cfg.data.img.img_step
        
        ## out dir 
        self.out_dir = osp.join(self.scans_files_dir, 'Auxiliary', self.model_name)
        common.ensure_dir(self.out_dir)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))
        
    def register_model(self):
        pass
        
    def inference(self, imgs_tensor):
        feature = self.model.backbone(imgs_tensor)[-1]
        return feature
        
    def generatePatchDepthScan(self, scan_id, depth_scale = 1000.0):
        # generate patch depth scan
        
        # load intrinsic
        color_info = scan3r.load_intrinsics(self.scans_scenes_dir, scan_id, 'color')
        color_intric = color_info['intrinsic_mat']
        depth_info = scan3r.load_intrinsics(self.scans_scenes_dir, scan_id, 'depth')
        depth_intric = depth_info['intrinsic_mat']
        affine_matrix_depth2color = color_intric @ np.linalg.inv(depth_intric)
        color_w, color_h = int(color_info['width']), int(color_info['height'])
        
        # load depth images 
        depth_paths = scan3r.load_depth_paths(self.scans_dir, scan_id, self.step)
        depth_imgs = {}
        for frame_idx, depth_path in depth_paths.items():
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
            depth_img_warped = cv2.warpPerspective(depth_img, affine_matrix_depth2color, 
                                        (color_w, color_h), flags=cv2.INTER_NEAREST)
            depth_imgs[frame_idx] = depth_img_warped
            
        if self.vis:
            plt.imshow(depth_imgs['000000'])
            plt.show()
        
        # load gt anno
        gt_anno_file = self.gt_anno_path[scan_id]
        gt_anno = common.load_pkl_data(gt_anno_file)
        # get patch anno
        patch_anno = self.patch_anno[scan_id]
        
        # calculate patch depth
        patch_position_frames = {}
        for frame_idx, depth_img in depth_imgs.items():
            patch_position = self.get_patch_position(depth_img, patch_anno[frame_idx], gt_anno[frame_idx], color_intric)
            patch_position_frames[frame_idx] = patch_position
        
        if self.vis:
            plt.imshow(patch_position_frames['000000'][:, :, 2])
            plt.show()
            
        return patch_position_frames

    def get_patch_position(self, depth_img, patch_anno, gt_anno, intrinsic, th = 0.2):
        patch_depth = np.zeros((self.image_patch_h, self.image_patch_w,3), dtype=np.float32)
        patch_size_h = self.image_h // self.image_patch_h
        patch_size_w = self.image_w // self.image_patch_w
        
        for h_i in range(self.image_patch_h):
            h_start, h_end = h_i * patch_size_h, (h_i + 1) * patch_size_w
            for w_i in range(self.image_patch_w):
                w_start, w_end = w_i * patch_size_w, (w_i + 1) * patch_size_w
                
                patch_obj_id = patch_anno[h_i, w_i]
                if patch_obj_id == self.undefined_obj_id:
                    continue
                
                depth_valid = depth_img[h_start:h_end, w_start:w_end] > 0.1
                obj_id_valid = gt_anno[h_start:h_end, w_start:w_end] == patch_obj_id
                valid_mask = np.logical_and(depth_valid, obj_id_valid)
                if valid_mask.sum() < th * patch_size_h * patch_size_w:
                    continue
                locs = np.where(valid_mask)
                depth_pxs_loc = [locs[0], locs[1]]
                depth_pxs_loc[0] += h_start
                depth_pxs_loc[1] += w_start
                # transform depth pixel to 3D point
                depth_pxs = np.stack([depth_pxs_loc[1], depth_pxs_loc[0]], axis=1)
                depth_pxs_aug = np.concatenate([depth_pxs, np.ones((depth_pxs.shape[0], 1))], axis=1)
                patch_points_unit = depth_pxs_aug @ np.linalg.inv(intrinsic).T
                patch_points = patch_points_unit * depth_img[depth_pxs_loc[0] , depth_pxs_loc[1]][:, None]
                # center point
                center_point = patch_points.mean(axis=0)
                patch_depth[h_i, w_i] = center_point
        return patch_depth
    
    def generatePatchPositions(self):
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                patch_depth_scan = self.generatePatchDepthScan(scan_id)
            out_file = osp.join(self.out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(patch_depth_scan, out_file)

def main():
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/auxiliary/depth/scan3r_patch_depth_generator.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_gcvit_generator = Scan3rPatchDepthGenerator(cfg, 'val', vis=False)
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generatePatchPositions()
    
if __name__ == "__main__":
    main()