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
from utils import common, scan3r, open3d, torch_util
from datasets.loaders import get_val_dataloader, get_train_dataloader

class Scan3rLidarClipDataset(data.Dataset):
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
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)

        # step
        self.step = self.cfg.data.img.img_step

        # scene_img_dir
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        # data type
        self.train_data_type = cfg.train.data_type

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
            
        # load 2D image indexs
        self.image_idxs = {}
        for scan_id in self.scan_ids:
            self.image_idxs[scan_id] = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id, self.step)
            
        # load 2D image features 
        self.feature_2D_folder_name = cfg.data.img_encoding.feature_dir
        self.img_features = {}
        for scan_id in self.scan_ids:
            img_features_scan_file = osp.join(self.scans_files_dir, self.feature_2D_folder_name, "{}.pkl".format(scan_id))
            self.img_features[scan_id] = common.load_pkl_data(img_features_scan_file)
            
        # load 2D depth image paths
        if self.train_data_type == 'depth':
            self.depth_image_paths = {}
            for scan_id in self.scan_ids:
                self.depth_image_paths[scan_id] = scan3r.load_depth_paths(self.scans_dir, scan_id, self.step)
            
        # load depth intrinsic matrix
        self.intrinsics = {}
        for scan_id in self.scan_ids:
            self.intrinsics[scan_id] = scan3r.load_intrinsics(
                self.scans_scenes_dir, scan_id, type='depth')['intrinsic_mat']
        self.depth_scale = self.cfg.data.depth.scale
        self.min_depth = self.cfg.data.depth.min_depth
        self.max_depth = self.cfg.data.depth.max_depth
        
        # load transform matrix from rescan to ref
        self.trans_rescan2ref = scan3r.read_transform_mat(scan_info_file) 
        
        # load 3D pointclouds
        if self.train_data_type == 'scan':
            self.loadScanPointClouds()
        elif self.train_data_type == 'depth':
            if self.split != 'train':
                self.loadScanPointClouds()
                self.loadDepthPointClouds()
        else:
            raise NotImplementedError
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()
        
        # fix candidate scan for val&test split for room retrieval
        self.num_scenes = cfg.data.cross_scene.num_scenes
        if self.split == 'val' or self.split == 'test':
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = self.sampleCandidateScenesForEachScan(scan_id, self.num_scenes)
        break_point = None
        
    def sampleCandidateScenesForEachScan(self, scan_id, num_scenes):
        candidate_scans = []
        scans_same_scene = self.refscans2scans[self.scans2refscans[scan_id]]
        # sample other scenes
        for scan in self.all_scans_split:
            if scan not in scans_same_scene:
                candidate_scans.append(scan)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        return sampled_scans
            
    def loadScanPointClouds(self):
        # load scan pcs
        self.scan_pcs = {}
        for scan_id in self.scan_ids:
            self.scan_pcs[scan_id] = scan3r.load_scan_pcs(
                    self.scans_scenes_dir, scan_id, self.trans_rescan2ref)
            
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
            
    def load_depthmap_pcs(self):
        return self.depthmap_pcs
    def load_scan_pcs(self):
        return self.scan_pcs
    
    
    def sampleCrossScenes(self, scan_id, num_scenes):
        candidate_scans = []
        scans_same_scene = self.refscans2scans[self.scans2refscans[scan_id]]
        # sample other scenes
        for scan in self.all_scans_split:
            if scan not in scans_same_scene:
                candidate_scans.append(scan)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        return sampled_scans
    
    def sampleCrossTime(self, scan_id):
        candidate_scans = []
        ref_scan = self.scans2refscans[scan_id]
        for scan in self.refscans2scans[ref_scan]:
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
                data_item = {}
                data_item['scan_id'] = scan_id
                data_item['frame_idx'] = frame_idx
                if self.train_data_type == 'scan':
                    pass
                elif self.train_data_type == 'depth':
                    if self.split != 'train':
                        if frame_idx in self.depthmap_pcs[scan_id]:
                            data_item['pc'] = self.depthmap_pcs[scan_id][frame_idx]
                    else:
                        data_item['depth_img_path'] = self.depth_image_paths[scan_id][frame_idx]
                        data_item['intrinsic'] = self.intrinsics[scan_id]
                else:
                    raise NotImplementedError
                
                data_items.append(data_item)
                    
        random.shuffle(data_items)
        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        return data_items
    
    def dataItem2DataDict(self, data_item):
        data_dict = {}
        
        # img info
        img_scan_id = data_item['scan_id']
        scan_id = data_item['scan_id']
        frame_idx = data_item['frame_idx']
        img_features = self.img_features[img_scan_id][frame_idx]
        
        # 3D info
        if self.train_data_type == 'scan':
            pass
        elif self.train_data_type == 'depth':
            if self.split != 'train':
                pcs = data_item['pc']
                data_dict['pc'] = pcs
            else:
                # load depth image
                depth_img_path = data_item['depth_img_path']
                depth_map = scan3r.load_depth_map(depth_img_path, self.depth_scale)
                ## trainsform depth image to point cloud with intrinsics
                intrinsic = data_item['intrinsic']
                point_cloud = scan3r.depthmap2pc(depth_map, intrinsic, [self.min_depth, self.max_depth])
                point_cloud = torch.from_numpy(point_cloud).float()
                data_dict['pc'] = point_cloud
        else:
            raise NotImplementedError

        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['img_feature'] = img_features
        
        
        if self.split != 'train':
            # sample across scenes for room retrieval
            candidate_scans = self.candidate_scans[scan_id]
            curr_scan = scan_id
            temporal_scan = self.sampleCrossTime(scan_id)
            # # load depth map pcs for room retrieval
            # candidate_depthmap_pcs = {}
            # for candidate_scan in candidate_scans:
            #     candidate_depthmap_pcs[candidate_scan] = self.depthmap_pcs[candidate_scan]
            # curr_scan_depthmap_pcs = self.depthmap_pcs[curr_scan]
            # temporal_scan_depthmap_pcs = self.depthmap_pcs[temporal_scan]
            # # load scan pcs for room retrieval
            # candidate_scan_pcs = {}
            # for candidate_scan in candidate_scans:
            #     candidate_scan_pcs[candidate_scan] = self.scan_pcs[candidate_scan]
            # curr_scan_pcs = self.scan_pcs[curr_scan]
            # temporal_scan_pcs = self.scan_pcs[temporal_scan]
            
            # # numpy to tensor
            # data_dict['candidate_depthmap_pcs'] = candidate_depthmap_pcs
            # data_dict['curr_scan_depthmap_pcs'] = curr_scan_depthmap_pcs
            # data_dict['temporal_scan_depthmap_pcs'] = temporal_scan_depthmap_pcs
            # data_dict['temporal_scan_id'] = temporal_scan
            
            # data_dict['candidate_scan_pcs'] = candidate_scan_pcs
            # data_dict['curr_scan_pcs'] = curr_scan_pcs
            # data_dict['temporal_scan_pcs'] = temporal_scan_pcs
            
            # save scan_id
            data_dict['candidate_scan_ids'] = candidate_scans
            data_dict['curr_scan_id'] = curr_scan
            data_dict['temporal_scan_id'] = temporal_scan
                
        return data_dict
    
    def collateBatchDicts(self, batch):
        
        
        # depth map pcs
        if self.train_data_type == 'depth':
            # filter our data item with no pcs
            batch_valid = []
            for data in batch:
                if data['pc'].shape[0] > 0:
                    batch_valid.append(data)
            batch = batch_valid
            data_dict['pcs_batch'] = [data['pc'].contiguous() for data in batch]
        
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
                if self.scans2refscans[batch[i]['scan_id']] == self.scans2refscans[batch[j]['scan_id']]:
                    e2j_matrix[i, j] = 0
                    e2j_matrix[j, i] = 0
        data_dict['e2j_matrix'] = torch.from_numpy(e2j_matrix).float()
        
        if self.split != 'train':
            # data_dict['candidate_depthmap_pcs_list'] = [data['candidate_depthmap_pcs'] for data in batch]
            # data_dict['curr_scan_depthmap_pcs_list'] = [data['curr_scan_depthmap_pcs'] for data in batch]
            # data_dict['temporal_scan_depthmap_pcs_list'] = [data['temporal_scan_depthmap_pcs'] for data in batch]
            # data_dict['temporal_scan_id_list'] = [data['temporal_scan_id'] for data in batch]
            # # load numpy pcs to tensor
            
            
            # data_dict['candidate_scan_pcs_list'] = [data['candidate_scan_pcs'] for data in batch]
            # data_dict['curr_scan_pcs_list'] = [data['curr_scan_pcs'] for data in batch]
            # data_dict['temporal_scan_pcs_list'] = [data['temporal_scan_pcs'] for data in batch]
        
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
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week8/lidar_clip_trainval_scan_r/scan3r_clip_train_scan_r.yaml"
    cfg = update_config(config, cfg_file, ensure_dir=False)
    # scan3r_ds = Scan3rLidarClipDataset(cfg, split='val')
    # print(len(scan3r_ds))
    # batch = [scan3r_ds[0], scan3r_ds[1], scan3r_ds[2]]
    # data_batch = scan3r_ds.collate_fn(batch)
    
    # visualize
    # vis = open3d.make_open3d_visualiser()
    # data_item = scan3r_ds[10]
    # # pcl = data_item['pc']
    # # print("point cloud of frame {} in scan {}".format(data_item['frame_idx'], data_item['scan_id']))
    # pcl = data_item['temporal_scan_pcs'][:,:3]
    # print("point cloud scan {}".format(data_item['scan_id']))
    
    # candidate_scans = list(data_item['candidate_scan_pcs'].keys())
    # candidate_scan = candidate_scans[0]
    # candidate_pcl = data_item['candidate_scan_pcs'][candidate_scan]
    # frame = '000005'
    # candidate_depthmap_pc = data_item['candidate_depthmap_pcs'][candidate_scan][frame]
    # print("point cloud of candidate scan {}, frame {} in scan {}".format(candidate_scan, frame, data_item['scan_id']))
    # pcl = candidate_depthmap_pc
    
    # pcd = open3d.make_open3d_point_cloud(pcl)
    # vis.add_geometry(pcd)
    # vis.run()
    
    # data_loder
    dataset_, val_dataloader = get_val_dataloader(cfg, Scan3rLidarClipDataset)
    total_iterations = len(val_dataloader)
    pbar = tqdm.tqdm(enumerate(val_dataloader), total=total_iterations)
    train_dataset, train_dataloader = get_train_dataloader(cfg, Scan3rLidarClipDataset)
    total_train_iterations = len(train_dataloader)
    breakpoint=None