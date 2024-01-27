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
from utils import common, scan3r, open3d, torch_util
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

class Scan3rLipLocDataset(data.Dataset):
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
            
        # load 2D image paths
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id, self.step)
        
        # img transform
        self.transform = get_transforms(self.split, self.cfg.data.img.img_size)
        
        # load transform matrix from rescan to ref
        self.trans_rescan2ref = scan3r.read_transform_mat(scan_info_file) 
        
        # load 3D range images pointclouds
        self.loadScansRangeImage()
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()
            
        break_point = None
            
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
            mesh_file = osp.join(self.scans_scenes_dir, scan_id, "labels.instances.annotated.v2.ply")
            proj_range, proj_color = scan3r.loadScanMeshRange(mesh_file, fov_up, fov_down, range_min, range_max, range_H, range_W)
            return proj_range, proj_color
        else:
            # load scan pcs
            pcs_with_color = scan3r.load_scan_pcs(
                    self.scans_scenes_dir, scan_id, self.trans_rescan2ref, color=True)
            pcs = pcs_with_color[:,:4]
            colors = pcs_with_color[:,4:]
            # project range image
            pcs_center  = np.mean(pcs, axis=0)
            proj_range, proj_color = scan3r.createRangeImage(
                pcs, colors, pcs_center, fov_up, fov_down, range_W, range_H, [range_min, range_max])
            return proj_range, proj_color
    
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
            for frame_idx in self.image_paths[scan_id]:
                data_item = {}
                data_item['scan_id'] = scan_id
                data_item['frame_idx'] = frame_idx
                data_item['img_path'] = self.image_paths[scan_id][frame_idx]
                data_items.append(data_item)
                    
        random.shuffle(data_items)
        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        return data_items
    
    def dataItem2DataDict(self, data_item):
        # img info
        scan_id = data_item['scan_id']
        frame_idx = data_item['frame_idx']
        
        img_path = data_item['img_path']
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) # type: uint8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1)) # channel first

        # 3D info
        range_image = self.range_imgs[scan_id]
        range_image = self.transform(image=range_image)['image']
        range_image = np.transpose(range_image, (2, 0, 1)) # channel first

        data_dict = {}
        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['range_image'] = range_image
        data_dict['img'] = img
        
        if self.split != 'train':
            # sample across scenes for room retrieval
            candidate_scans = self.sampleCrossScenes(scan_id, self.cfg.data.cross_scene.num_scenes)
            curr_scan = scan_id
            temporal_scan = self.sampleCrossTime(scan_id)
            
            # save scan_id
            data_dict['candidate_scan_ids'] = candidate_scans
            data_dict['curr_scan_id'] = curr_scan
            data_dict['temporal_scan_id'] = temporal_scan
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
                if self.scans2refscans[batch[i]['scan_id']] == self.scans2refscans[batch[j]['scan_id']]:
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
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week9/trainval_liploc_360_mesh/scan3r_liploc_train.yaml"
    cfg = update_config(config, cfg_file, ensure_dir=False)
    
    # dataset
    dataset = Scan3rLipLocDataset(cfg, 'train')
    
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    
    # # data_loder
    # dataset_, val_dataloader = get_val_dataloader(cfg, Scan3rLipLocDataset)
    # total_iterations = len(val_dataloader)
    # pbar = tqdm.tqdm(enumerate(val_dataloader), total=total_iterations)
    # for iteration, data_dict in pbar:
    #     pass
    breakpoint=None