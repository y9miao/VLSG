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

class Scan3rOpen3DDataset(data.Dataset):
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
            
        # load 2D image features path
        self.feature_2D_folder_name = cfg.data.img_encoding.feature_dir
        self.img_features_path = {}
        for scan_id in self.scan_ids:
            img_features_scan_file = osp.join(self.scans_files_dir, self.feature_2D_folder_name, "{}.pkl".format(scan_id))
            self.img_features_path[scan_id] = img_features_scan_file
            
        # load 3D obj embeddings and info
        self.feat_dim = cfg.data.feat_dim
        self.feature_3D_folder_name = cfg.data.obj_img.name
        self.obj_3D_embeddings = {}
        self.obj_3D_ids = {}
        for scan_id in self.scan_ids:
            obj_3D_embeddings_scan_file = osp.join(self.scans_files_dir, self.feature_3D_folder_name, "{}.pkl".format(scan_id))
            obj_3D_embeddings_scan = common.load_pkl_data(obj_3D_embeddings_scan_file)['obj_visual_emb']
            # average obj embeddings over frames
            obj_3D_embeddings_scan_ave = []
            self.obj_3D_ids[scan_id] = []
            for obj_id in obj_3D_embeddings_scan:
                obj_embs_list = []
                for frame_idx in obj_3D_embeddings_scan[obj_id]:
                    obj_embs_list.append(obj_3D_embeddings_scan[obj_id][frame_idx].reshape(1, -1))
                if len(obj_embs_list) > 0:
                    obj_3D_embeddings_cat = np.concatenate(obj_embs_list, axis=0)
                    obj_3D_embeddings_scan_ave.append(np.mean(obj_3D_embeddings_cat, axis=0))
                else:
                    obj_3D_embeddings_scan_ave.append(np.ones((self.feat_dim)))
                self.obj_3D_ids[scan_id].append(obj_id)
            obj_3D_embeddings_scan_ave = np.array(obj_3D_embeddings_scan_ave)
            # normalize obj embeddings
            obj_3D_embeddings_scan_ave = obj_3D_embeddings_scan_ave / np.linalg.norm(obj_3D_embeddings_scan_ave, axis=1, keepdims=True)
            self.obj_3D_embeddings[scan_id] = obj_3D_embeddings_scan_ave
            
            self.obj_3D_ids[scan_id] = np.array(self.obj_3D_ids[scan_id])
            
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
                data_items.append(data_item)
        if self.split == 'train':
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
        img_features_path = self.img_features_path[img_scan_id]

        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['img_feature'] = common.load_pkl_data(img_features_path)[frame_idx]
        
        
        if self.split != 'train':
            # sample across scenes for room retrieval
            candidate_scans = self.candidate_scans[scan_id]
            temporal_scan = self.sampleCrossTime(scan_id)
            
            # save scan_id
            data_dict['candidate_scan_ids'] = candidate_scans
            data_dict['temporal_scan_id'] = temporal_scan
                
        return data_dict
    
    def collateBatchDicts(self, batch):
        
        batch_size = len(batch)
        data_dict = {}
        data_dict['batch_size'] = batch_size
        # frame info 
        data_dict['scan_ids'] = [data['scan_id'] for data in batch]
        data_dict['frame_idxs'] = [data['frame_idx'] for data in batch]
        img_features_batch = np.stack([data['img_feature'] for data in batch]) # (B, D)
        data_dict['img_features'] = img_features_batch# (B, D)
        assert img_features_batch.shape[0] == batch_size
        
        if self.split != 'train':
            data_dict['candidate_scan_ids_list'] = [data['candidate_scan_ids'] for data in batch]
            data_dict['temporal_scan_id_list'] = [data['temporal_scan_id'] for data in batch]
            
            # get all scans involved
            scans_involoved = set()
            scans_involoved.update(data_dict['scan_ids'])
            for candidate_scan_ids in data_dict['candidate_scan_ids_list']:
                scans_involoved.update(candidate_scan_ids)
            for temporal_scan_id in data_dict['temporal_scan_id_list']:
                if temporal_scan_id is not None:
                    scans_involoved.add(temporal_scan_id)
            # get 3D obj embeddings for those scans
            obj_3D_embeddings = {}
            for scan_id in scans_involoved:
                obj_3D_embeddings[scan_id] = self.obj_3D_embeddings[scan_id]
            data_dict['obj_3D_embeddings'] = obj_3D_embeddings
            # obj_3D_ids
            obj_3D_ids = {}
            for scan_id in scans_involoved:
                obj_3D_ids[scan_id] = self.obj_3D_ids[scan_id]
            data_dict['obj_3D_ids'] = obj_3D_ids
        
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
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/src/room_retrieval/OpenMask3D/openmask3D_retrieval.yaml"
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
    dataset_, val_dataloader = get_val_dataloader(cfg, Scan3rOpen3DDataset)
    total_iterations = len(val_dataloader)
    pbar = tqdm.tqdm(enumerate(val_dataloader), total=total_iterations)
    # train_dataset, train_dataloader = get_train_dataloader(cfg, Scan3rOpen3DDataset)
    # total_train_iterations = len(train_dataloader)
    for i, data in pbar:
        print(i)
        pass
    
    breakpoint=None