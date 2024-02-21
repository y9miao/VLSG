import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data
import cv2
import sys
import tqdm
import albumentations as A

from yaml import scan
dataset_dir = osp.dirname(osp.abspath(__file__))
src_dir = osp.dirname(dataset_dir)
sys.path.append(src_dir)
from utils import common, scan3r, open3d, torch_util, scannet_utils
from datasets.loaders import get_val_dataloader

def get_transforms(mode, size):
    if mode == "train":
        return A.Compose(
            [   
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
                A.CoarseDropout(always_apply=False, p=0.9, max_holes=5, max_height=100, max_width=100, min_height=50, min_width=50, fill_value=(0, 0, 0), mask_fill_value=None),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

class ScannetLipLocDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
        # set random seed
        self.seed = cfg.seed
        random.seed(self.seed)
        
        # sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        
        # scannet scans info
        self.split = split
        scans_info_file = osp.join(cfg.data.root_dir, 'files', 'scans_{}.pkl'.format(split))
        self.rooms_info = common.load_pkl_data(scans_info_file)
        self.scan_ids = []
        self.scan2room = {}
        for room_id in self.rooms_info:
            self.scan_ids += self.rooms_info[room_id]
            for scan_id in self.rooms_info[room_id]:
                self.scan2room[scan_id] = room_id
        self.room2scans = self.rooms_info
        self.scans_files_dir = osp.join(cfg.data.root_dir, 'files')

        # step
        self.step = self.cfg.data.img.img_step
            
        # load 2D image indexs
        self.step = self.cfg.data.img.img_step
        self.root_dir = cfg.data.root_dir
        self.split_folder = osp.join(self.root_dir, self.split)
        self.image_idxs = {}
        self.image_paths = {}   
        for scan_id in self.scan_ids:
            self.image_idxs[scan_id] = scannet_utils.load_frame_idxs(self.split_folder, scan_id, self.step)
            self.image_paths[scan_id] = scannet_utils.load_frame_paths(self.split_folder, scan_id, self.step)
            
        # load patch anno
        self.scans_files_dir = osp.join(self.root_dir, 'files')
        self.patch_anno = {}
        patch_anno_name = cfg.data.gt_patch
        patch_anno_th = cfg.data.gt_patch_th
        self.patch_anno_folder = osp.join(self.scans_files_dir, patch_anno_name)
        for scan_id in self.scan_ids:
            patch_anno_scan = common.load_pkl_data(osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id)))
            self.patch_anno[scan_id] = {}
            # filter frames without enough patches
            for frame_idx in self.image_idxs[scan_id]:
                if frame_idx in patch_anno_scan:
                    num_valid_patches = np.sum(patch_anno_scan[frame_idx]!=0)
                    if num_valid_patches * 1.0 / patch_anno_scan[frame_idx].size > patch_anno_th:
                        self.patch_anno[scan_id][frame_idx] = patch_anno_scan[frame_idx]
            
        # img transform
        self.transform = get_transforms(self.split, self.cfg.data.img.img_size)
        
        # load 3D range images pointclouds
        self.loadScansRangeImage()
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()

        # fix candidate scan for val&test split for room retrieval
        self.num_scenes = cfg.data.cross_scene.num_scenes
        if self.split == 'val' or self.split == 'test':
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = scannet_utils.sampleCandidateScenesForEachScan(
                    scan_id, self.scan_ids, self.rooms_info, self.scan2room, self.num_scenes)
        
    def sampleCandidateScenesForEachScan(self, scan_id, num_scenes):
        candidate_scans = []
        scans_same_scene = self.room2scans[self.scan2room[scan_id]]
        # sample other scenes
        for scan in self.scan_ids:
            if scan not in scans_same_scene:
                candidate_scans.append(scan)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        return sampled_scans
            
    def loadScansRangeImage(self):
        
        use_saved_range = self.cfg.data.pointcloud.use_saved_range
        save_range = self.cfg.data.pointcloud.save_range
        range_name = self.cfg.data.pointcloud.range_name
        self.range_folder = osp.join(self.scans_files_dir, range_name)
        self.range_folder_depth = osp.join(self.range_folder, 'depth')
        self.range_folder_color = osp.join(self.range_folder, 'color')
        if save_range:
            common.ensure_dir(self.range_folder_depth)
            common.ensure_dir(self.range_folder_color)
        
        self.range_imgs = {}
        self.range_folder = osp.join(self.scans_files_dir, range_name)
        if use_saved_range:
            # load range pcs
            for scan_id in self.scan_ids:
                range_depth_file = osp.join(self.range_folder_depth, scan_id+'.png')
                self.range_imgs[scan_id] = cv2.imread(range_depth_file, cv2.IMREAD_UNCHANGED)
        else:
            fov_up = self.cfg.data.pointcloud.fov_up
            fov_down = self.cfg.data.pointcloud.fov_down
            range_min = self.cfg.data.pointcloud.range_min 
            range_max = self.cfg.data.pointcloud.range_max
            range_W = self.cfg.data.pointcloud.range_W
            range_H = self.cfg.data.pointcloud.range_H
            for scan_id in self.scan_ids:
                proj_range, proj_color = self.loadScanRange(scan_id, fov_up, fov_down, range_min, range_max, range_W, range_H)
                # record range image
                self.range_imgs[scan_id] = proj_range
                if save_range:
                    # save range image
                    range_depth_file = osp.join(self.range_folder_depth, scan_id+'.png')
                    cv2.imwrite(range_depth_file, proj_range)
                    # save color image
                    range_color_file = osp.join(self.range_folder_color, scan_id+'.png')
                    proj_color_bgr = cv2.cvtColor(proj_color, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(range_color_file, proj_color_bgr)
    
    def loadScanRange(self, scan_id, fov_up, fov_down, range_min, range_max, range_W, range_H):
        if self.cfg.data.pointcloud.use_mesh:
            mesh_file = scannet_utils.load_mesh_path(self.split_folder, scan_id, "{}_vh_clean_2.labels.ply".format(scan_id))
            proj_range, proj_color = scan3r.loadScanMeshRange(mesh_file, fov_up, fov_down, range_min, 
                                                              range_max, range_H, range_W)
            return proj_range, proj_color
        else:
            raise NotImplementedError("Not implemented for pointcloud")
    
    def sampleCrossRooms(self, scan_id):
        candidate_scans = []
        room_id = self.scan2room[scan_id]
        for scan in self.rooms_info[room_id]:
            if scan != scan_id:
                candidate_scans.append(scan)
        if len(candidate_scans) == 0:
            return None
        else:
            sampled_scan = random.sample(candidate_scans, 1)[0]
            return sampled_scan
    
    def generateDataItems(self):
        data_items = []
        # iterate over scans
        for scan_id in self.scan_ids:
            # sample rescan
            if self.split == 'train':
                for frame_idx in self.image_paths[scan_id]:
                    if frame_idx not in self.patch_anno[scan_id]:
                        continue
                    data_item = {}
                    data_item['scan_id'] = scan_id
                    data_item['rescan_id'] = None
                    data_item['frame_idx'] = frame_idx
                    data_item['img_path'] = self.image_paths[scan_id][frame_idx]
                    data_items.append(data_item)
            else:
                rescan_id = self.sampleCrossRooms(scan_id)
                if rescan_id is None:
                    continue
                # iterate over images
                for frame_idx in self.image_paths[scan_id]:
                    if frame_idx not in self.patch_anno[scan_id]:
                        continue
                    data_item = {}
                    data_item['scan_id'] = scan_id
                    data_item['rescan_id'] = rescan_id
                    data_item['frame_idx'] = frame_idx
                    data_item['img_path'] = self.image_paths[scan_id][frame_idx]
                    data_items.append(data_item)
                    
        random.shuffle(data_items) if self.split == 'train' else None
        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        return data_items
    
    def dataItem2DataDict(self, data_item):
        # img info
        scan_id = data_item['scan_id']
        rescan_id = data_item['rescan_id']
        frame_idx = data_item['frame_idx']
        
        img_path = data_item['img_path']
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) # type: uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)) # channel first

        # 3D info
        range_image = self.range_imgs[scan_id] if self.split == 'train' else self.range_imgs[rescan_id].copy()
        range_image = self.transform(image=range_image)['image']
        range_image = np.transpose(range_image, (2, 0, 1)) # channel first

        data_dict = {}
        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['range_image'] = range_image
        data_dict['img'] = img
        
        # sample across scenes for room retrieval
        if self.split != 'train':
            candidate_scans = self.candidate_scans[scan_id]
            curr_scan = scan_id
            # save scan_id
            data_dict['candidate_scan_ids'] = candidate_scans
            data_dict['curr_scan_id'] = curr_scan
            data_dict['temporal_scan_id'] = rescan_id

        return data_dict
    
    def getRangeImagesTensor(self, scan_id_list):
        scan_range_images = []
        for scan_id in scan_id_list:
            range_image = self.range_imgs[scan_id]
            range_image = self.transform(image=range_image)['image']
            range_image = np.transpose(range_image, (2, 0, 1))
            scan_range_images.append(range_image)
        range_image = np.stack(scan_range_images)
        range_image_tensor = torch.from_numpy(range_image).float() # (B, 3, H, W)
        return range_image_tensor
    
    def collateBatchDicts(self, batch):  
        batch_size = len(batch)
        data_dict = {}
        data_dict['batch_size'] = batch_size
        # frame info 
        data_dict['scan_ids'] = [data['scan_id'] for data in batch]
        data_dict['frame_idxs'] = [data['frame_idx'] for data in batch]
        
        imgs_batch = np.stack([data['img'] for data in batch]) 
        data_dict['camera_image'] = torch.from_numpy(imgs_batch).float() # (B, 3, H, W)
        # 3d info
        range_imgs_batch = np.stack([data['range_image'] for data in batch])
        data_dict['lidar_image'] = torch.from_numpy(range_imgs_batch).float() # (B, 3, H, W)
        
        # e2j_matrix for unpaired data
        e2j_matrix = np.ones((batch_size, batch_size))
        np.fill_diagonal(e2j_matrix, 0)
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if self.scan2room[batch[i]['scan_id']] == self.scan2room[batch[j]['scan_id']]:
                    e2j_matrix[i, j] = 0
                    e2j_matrix[j, i] = 0
        data_dict['e2j_matrix'] = torch.from_numpy(e2j_matrix).float()
        if self.split != 'train':
            data_dict['candidate_scan_ids_list'] = [data['candidate_scan_ids'] for data in batch]
            data_dict['curr_scan_id_list'] = [data['curr_scan_id'] for data in batch]
            data_dict['temporal_scan_id_list'] = [data['temporal_scan_id'] for data in batch]
        if len(batch) > 0:
            return data_dict
        else:
            return None
    
    def __getitem__(self, idx):
        data_dict = self.dataItem2DataDict(self.data_items[idx])
        return data_dict
    
    def collate_fn(self, batch):
        data_dict = self.collateBatchDicts(batch)
        return data_dict
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week12/trainval_liploc/scannet_liploc.yaml"
    cfg = update_config(config, cfg_file, ensure_dir=False)
    
    # dataset
    dataset = ScannetLipLocDataset(cfg, 'val')
    
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    
    # # data_loder
    # dataset_, val_dataloader = get_val_dataloader(cfg, ScannetLipLocDataset)
    # total_iterations = len(val_dataloader)
    # pbar = tqdm.tqdm(enumerate(val_dataloader), total=total_iterations)
    # for iteration, data_dict in pbar:
    #     pass
    breakpoint=None