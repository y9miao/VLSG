from math import e
import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data
import cv2
import sys
import tqdm

from yaml import scan
dataset_dir = osp.dirname(osp.abspath(__file__))
src_dir = osp.dirname(dataset_dir)
sys.path.append(src_dir)
from utils import common, scan3r, open3d, torch_util, scannet_utils
from datasets.loaders import get_val_dataloader, get_train_dataloader

class ScannetLidarClipDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
        # set random seed
        self.seed = cfg.seed
        random.seed(self.seed)
        
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
            
        # load 2D image indexs
        self.step = self.cfg.data.img.img_step
        self.root_dir = cfg.data.root_dir
        self.split_folder = osp.join(self.root_dir, self.split)
        self.image_idxs = {}
        for scan_id in self.scan_ids:
            self.image_idxs[scan_id] = scannet_utils.load_frame_idxs(self.split_folder, scan_id, self.step)
        # load 2D image features paths
        self.feature_2D_folder_name = cfg.data.img_encoding.feature_dir
        self.feature_folder = osp.join(self.root_dir, 'files', self.feature_2D_folder_name)
        self.features_path = {}
        for scan_id in self.scan_ids:
            self.features_path[scan_id] = {}
            for frame_idx in self.image_idxs[scan_id]:
                feature_file = osp.join(self.feature_folder, scan_id, "{}.npy".format(frame_idx))
                self.features_path[scan_id][frame_idx] = feature_file
                
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
        
        # load 3D pointclouds
        # data type
        self.train_data_type = cfg.train.data_type
        if self.train_data_type == 'scan':
            self.loadScanPointClouds()
        else:
            raise NotImplementedError
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()
        
        # fix candidate scan for val&test split for room retrieval
        self.num_scenes = cfg.data.cross_scene.num_scenes
        if self.split == 'val' or self.split == 'test':
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = scannet_utils.sampleCandidateScenesForEachScan(
                    scan_id, self.scan_ids, self.rooms_info, self.scan2room, self.num_scenes)
        break_point = None
            
    def loadScanPointClouds(self):
        # load scan pcs
        self.scan_pcs = {}
        for scan_id in self.scan_ids:
            self.scan_pcs[scan_id] = scannet_utils.load_scan_pcs(self.split_folder, scan_id)
            # transform to tensor
            self.scan_pcs[scan_id] = torch.from_numpy(self.scan_pcs[scan_id]).float()
    
    def loadDepthPointClouds(self):
        # load depth map pcs
        self.depthmap_pcs = {}
        for scan_id in self.scan_ids:
            self.depthmap_pcs[scan_id] = scan3r.load_scan_depth_pcs(
                    self.scans_dir, scan_id, self.depth_scale, 
                    self.intrinsics[scan_id], [self.min_depth, self.max_depth], self.step)
            # transform to tensor
            self.depthmap_pcs[scan_id] = {
                frame_idx: torch.from_numpy(self.depthmap_pcs[scan_id][frame_idx]).float()
                for frame_idx in self.depthmap_pcs[scan_id]
            }
            
    def load_scan_pcs(self):
        return self.scan_pcs
    
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
            # obj_3D_embeddings_scan = self.obj_3D_embeddings[scan_id]
            # iterate over images
            for frame_idx in self.image_idxs[scan_id]:
                if frame_idx not in self.patch_anno[scan_id]:
                    continue
                data_item = {}
                if self.split == 'train':
                    data_item['scan_id'] = scan_id
                    data_item['rescan_id'] = None
                    data_item['frame_idx'] = frame_idx
                else:
                    rescan_id = self.sampleCrossRooms(scan_id)
                    if rescan_id is None:
                        continue
                    data_item['scan_id'] = scan_id
                    data_item['rescan_id'] = rescan_id
                    data_item['frame_idx'] = frame_idx
                data_items.append(data_item)
                    
        random.shuffle(data_items) if self.split == 'train' else None
        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        return data_items
    
    def dataItem2DataDict(self, data_item):
        # img info
        scan_id = data_item['scan_id']
        frame_idx = data_item['frame_idx']
        rescan_id = data_item['rescan_id']
        # load 2D features
        img_features = np.load(self.features_path[scan_id][frame_idx])
        
        data_dict = {}
        data_dict['scan_id'] = scan_id if self.split == 'train' else rescan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['img_feature'] = img_features
        
        if self.split != 'train':
            # sample across scenes for room retrieval
            candidate_scans = self.candidate_scans[scan_id]
            curr_scan = scan_id
            # save scan_id
            data_dict['candidate_scan_ids'] = candidate_scans
            data_dict['curr_scan_id'] = curr_scan
            data_dict['temporal_scan_id'] = scan_id
        return data_dict
    
    def collateBatchDicts(self, batch):
        
        batch_size = len(batch)
        data_dict = {}
        data_dict['batch_size'] = batch_size
        # frame info 
        data_dict['scan_ids'] = [data['scan_id'] for data in batch]
        data_dict['frame_idxs'] = [data['frame_idx'] for data in batch]
        img_features_batch = np.stack([data['img_feature'] for data in batch]) # (B, D)
        data_dict['img_features'] = torch.from_numpy(img_features_batch).float() # (B, D)
        assert img_features_batch.shape[0] == batch_size

        # e2j_matrix for unpaired data
        e2j_matrix = np.ones((batch_size, batch_size))
        np.fill_diagonal(e2j_matrix, 0)
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if batch[i] == None or batch[j] == None:
                    break_point = None
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
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week12/trainval_lidarclip/scannet_lidarclip.yaml"
    cfg = update_config(config, cfg_file, ensure_dir=False)
    # vis.run()
    
    # data_loder
    # dataset_, val_dataloader = get_val_dataloader(cfg, ScannetLidarClipDataset)
    # total_iterations = len(val_dataloader)
    # pbar = tqdm.tqdm(enumerate(val_dataloader), total=total_iterations)
    # for i, data in pbar:
    #     pass
    
    train_dataset, train_dataloader = get_train_dataloader(cfg, ScannetLidarClipDataset)
    total_train_iterations = len(train_dataloader)
    pbar = tqdm.tqdm(enumerate(train_dataloader), total=total_train_iterations)
    for i, data in pbar:
        pass
    breakpoint=None