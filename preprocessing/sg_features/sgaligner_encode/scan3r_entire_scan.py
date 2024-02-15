import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
from configs import config, update_config
import sys
sys.path.append('..')

from utils import common, scan3r
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='3R Scan configuration file name')
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to run subscan generation on')
    parser.add_argument('--rescan', dest='rescan', default=False, action='store_true', help='get ref scan or rescan')

    args = parser.parse_args()
    return parser, args

class Scan3REntireScanDataset(data.Dataset):
    def __init__(self, cfg, split, rescan):
        self.split = split
        self.use_predicted = cfg.use_predicted
        self.pc_resolution = cfg.val.pc_res if split == 'val' else cfg.train.pc_res
        self.model_name = cfg.model_name
        self.scan_type = cfg.scan_type
        self.data_root_dir = cfg.data.root_dir
        
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname

        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        
        self.mode = 'orig' if self.split == 'train' else cfg.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
        
        rescan_note = ''
        if rescan:
            rescan_note = 're'
        self.scan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, rescan_note)), dtype=str)
        
        self.is_training = self.split == 'train'
        self.do_augmentation = False if self.split == 'val' else cfg.train.use_augmentation

        self.rot_factor = cfg.train.rot_factor
        self.augment_noise = cfg.train.augmentation_noise

        # Jitter
        self.scale = 0.01
        self.clip = 0.05

        # Random Rigid Transformation
        self._rot_mag = 45.0
        self._trans_mag = 0.5
        
    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        
        scan_id = self.scan_ids[idx]
        # Centering
        points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(scan_id)), obj_ids = None)
        pcl_center = np.mean(points, axis=0)

        data_dict = common.load_pkl_data(osp.join(self.scans_files_dir_mode, 'data/{}.pkl'.format(scan_id)))
        
        object_ids = data_dict['objects_id']
        global_object_ids = data_dict['objects_cat']
        edges = data_dict['edges']
        object_points = data_dict['obj_points'][self.pc_resolution] - pcl_center
        object_id2idx = data_dict['object_id2idx']

        object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
        edges = torch.from_numpy(edges)
        if not self.use_predicted:
            bow_vec_obj_attr_feats = torch.from_numpy(data_dict['bow_vec_object_attr_feats'])
        
        else:
            bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], 41)
        
        bow_vec_obj_edge_feats = torch.from_numpy(data_dict['bow_vec_object_edge_feats'])
        rel_pose = torch.from_numpy(data_dict['rel_trans'])

        data_dict = {} 
        data_dict['obj_ids'] = object_ids
        data_dict['tot_obj_pts'] = object_points
        data_dict['graph_per_obj_count'] = np.array([object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([edges.shape[0]])
        
        data_dict['tot_obj_count'] = object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = rel_pose
        data_dict['edges'] = edges    

        data_dict['global_obj_ids'] = global_object_ids
        data_dict['scene_ids'] = [scan_id]        
        data_dict['pcl_center'] = pcl_center
        
        return data_dict

    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def collate_fn(self, batch):
        tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        tot_bow_vec_object_attr_feats = self._collate_feats(batch, 'tot_bow_vec_object_attr_feats')
        tot_bow_vec_object_edge_feats = self._collate_feats(batch, 'tot_bow_vec_object_edge_feats')    
        tot_rel_pose = self._collate_feats(batch, 'tot_rel_pose')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in batch])
        data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_object_attr_feats.double()
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_object_edge_feats.double()
        data_dict['tot_rel_pose'] = tot_rel_pose.double()
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in batch])
        data_dict['graph_per_edge_count'] = np.stack([data['graph_per_edge_count'] for data in batch])
        data_dict['edges'] = self._collate_feats(batch, 'edges')
        data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in batch])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
        data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in batch])
        data_dict['batch_size'] = len(batch)

        return data_dict
        
if __name__ == '__main__':
    _, args = parse_args()
    cfg = update_config(config, args.config, ensure_dir=False)
    scan3r_ds = Scan3REntireScanDataset(cfg, split=args.split, rescan=args.rescan)
    print("total {} scenes in {} set".format(len(scan3r_ds), args.split) )
    # for data_item in scan3r_ds:
    #     print(data_item['scene_ids'])    