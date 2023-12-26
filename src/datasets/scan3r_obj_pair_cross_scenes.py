import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data
import cv2
import sys
sys.path.append('..')
sys.path.append('../..')
from utils import common, scan3r

class PatchObjectPairCrossScenesDataSet(data.Dataset):
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
        # 3D_obj_embedding_dir
        self.scans_3Dobj_embeddings_dir = osp.join(self.scans_files_dir_mode, "embeddings")
        # 2D images
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        # 2D_patch_anno_dir
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.step = self.cfg.data.img.img_step
        # self.patch_w_size_int = int(self.image_w / self.image_patch_w)
        # self.patch_h_size_int = int(self.image_h / self.image_patch_h)
        self.num_patch = self.image_patch_w * self.image_patch_h
        self.patch_anno_folder_name = "patch_anno_{}_{}".format(self.image_patch_w, self.image_patch_h)
        self.scans_2Dpatch_anno_dir = osp.join(self.scans_files_dir, "patch_anno", self.patch_anno_folder_name)
        # scene_img_dir
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        # cross scenes cfg
        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes
        self.num_negative_samples = cfg.data.cross_scene.num_negative_samples
        
        # scans info
        self.scan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_scans.txt'.format(split)), dtype=str)
        self.rescan = cfg.data.rescan
        rescan_note = ''
        if self.rescan:
            rescan_note = 're'
            rescan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, rescan_note)), dtype=str)
            self.scan_ids = np.concatenate([self.scan_ids, rescan_ids])
            
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
                    self.scans2refscans[scan] = ref_scan_id
            
        # load 2D image paths
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id, self.step)
            
        # load 3D obj embeddings
        self.obj_3D_embeddings = {}
        for scan_id in self.scan_ids:
            embedding_file = osp.join(self.scans_3Dobj_embeddings_dir, "{}.pkl".format(scan_id))
            embeddings = common.load_pkl_data(embedding_file)
            obj_3D_embeddings_scan = embeddings['obj_embeddings']
            self.obj_3D_embeddings[scan_id] = obj_3D_embeddings_scan
            
        # load 3D obj semantic annotations
        self.obj_3D_anno = {}
        self.objs_config_file = osp.join(self.scans_files_dir, 'objects.json')
        objs_configs = common.load_json(self.objs_config_file)['scans']
        scans_objs_info = {}
        for scan_item in objs_configs:
            scan_id = scan_item['scan']
            objs_info = scan_item['objects']
            scans_objs_info[scan_id] = objs_info
        for scan_id in self.scan_ids:
            self.obj_3D_anno[scan_id] = {}
            for obj_item in scans_objs_info[scan_id]:
                obj_id = int(obj_item['id'])
                obj_nyu_category = int(obj_item['nyu40'])
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, obj_nyu_category)
            
        # load 2D patch annotation
        self.obj_2D_patch_anno = {}
        for scan_id in self.scan_ids:
            patch_anno_scan_file = osp.join(self.scans_2Dpatch_anno_dir, "{}.pkl".format(scan_id))
            self.obj_2D_patch_anno[scan_id] = common.load_pkl_data(patch_anno_scan_file)
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()
        
    # sample objects and other scenes for each data item 
    def sampleCrossScenes(self, scan_id, num_scenes, num_objects):
        
        candidate_scans = []
        scans_same_scene = self.refscans2scans[self.scans2refscans[scan_id]]
        # sample other scenes
        for scan_id in self.scan_ids:
            if scan_id not in scans_same_scene:
                candidate_scans.append(scan_id)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        
        # sample objects of sampled scenes
        candidate_objs = []
        for sampled_scan_id in sampled_scans:
            # record scan_id, obj_id, category for each candidate object
            # some obj may not have embeddings, thus only sample from objs with embeddings
            candidate_objs += [self.obj_3D_anno[sampled_scan_id][obj_id] 
                               for obj_id in self.obj_3D_embeddings[sampled_scan_id]]
        sampled_objs = []
        if num_objects > 0:
            if num_objects < len(candidate_objs):
                sampled_objs = random.sample(candidate_objs, num_objects)
            else:
                # sample all objects if not enough objects
                sampled_objs = candidate_objs
        else:
            sampled_objs = candidate_objs
        return sampled_objs
            
    def generateDataItems(self):
        data_items = []
        # iterate over scans
        for scan_id in self.scan_ids:
            # obj_3D_embeddings_scan = self.obj_3D_embeddings[scan_id]
            # iterate over images
            obj_2D_patch_anno_scan = self.obj_2D_patch_anno[scan_id]
            image_paths = self.image_paths[scan_id]
            for frame_idx in image_paths:
                data_item_dict = {}
                # 2D info
                data_item_dict['img_path'] = image_paths[frame_idx]
                data_item_dict['patch_anno'] = obj_2D_patch_anno_scan[frame_idx]
                data_item_dict['frame_idx'] = frame_idx
                # 3D info
                data_item_dict['scan_id'] = scan_id
                data_items.append(data_item_dict)
                # sample cross scenes
                if self.use_cross_scene:
                    sampled_objs = self.sampleCrossScenes(scan_id, self.num_scenes, self.num_negative_samples)
                    data_item_dict['obj_across_scenes'] = sampled_objs
                else:
                    data_item_dict['obj_across_scenes'] = []
                
        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        
        return data_items
    
    def __getitem__(self, idx):
        data_item = self.data_items[idx]

        # 3D object embeddings
        scan_id = data_item['scan_id']
        obj_3D_embeddings = self.obj_3D_embeddings[scan_id]
        obj_3D_embeddings_arr = np.array([obj_3D_embeddings[obj_id] for obj_id in obj_3D_embeddings])
        num_objs = len(obj_3D_embeddings)
        num_objs_across_scenes = len(data_item['obj_across_scenes'])
        obj_3D_id2idx = {} # only for objs in current scene
        obj_3D_idx2info = {} # for objs in current scene and other scenes
        idx = 0
        for obj_id in  obj_3D_embeddings:
            obj_3D_id2idx[obj_id] = idx
            obj_3D_idx2info[idx] = self.obj_3D_anno[scan_id][obj_id]
            idx += 1
            
        obj_3D_across_scnes_embeddings = []
        for obj_info in data_item['obj_across_scenes']:
            obj_3D_idx2info[idx] = obj_info
            scan_id_across_scenes = obj_info[0]
            obj_id_across_scenes = obj_info[1]
            obj_3D_across_scnes_embeddings.append(
                self.obj_3D_embeddings[scan_id_across_scenes][obj_id_across_scenes])
            idx += 1
            
        obj_3D_across_scnes_embeddings_arr = np.array(obj_3D_across_scnes_embeddings)
        if num_objs_across_scenes > 0:
            obj_3D_embeddings_arr = np.concatenate(
                [obj_3D_embeddings_arr, obj_3D_across_scnes_embeddings_arr], axis=0)
            
        # img data
        img_path = data_item['img_path']
        frame_idx = data_item['frame_idx']
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) # type: ignore
        img = cv2.resize(img, (self.image_resize_w, self.image_resize_h),  # type: ignore
                         interpolation=cv2.INTER_LINEAR) # type: ignore
        # 2D gt obj anno
        obj_2D_patch_anno = data_item['patch_anno']
        
        # rotate images and annotations
        patch_h = self.image_patch_h
        patch_w = self.image_patch_w
        if self.img_rotate:
            img = img.transpose(1, 0, 2)
            img = np.flip(img, 1)
            obj_2D_patch_anno = obj_2D_patch_anno.transpose(1, 0)
            obj_2D_patch_anno = np.flip(obj_2D_patch_anno, 1)
            patch_h = self.image_patch_w
            patch_w = self.image_patch_h
            
        obj_2D_patch_anno_flatten = obj_2D_patch_anno.reshape(-1)
        
        # generate relationship matrix for contrast learning 
        ## From 2D to 3D, denote as e1i_matrix, e1j_matrix, e2j_matrix      
        ## e1i_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object pairs
        ## e2j_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object unpairs
        e1i_matrix = np.zeros( (self.num_patch, num_objs+num_objs_across_scenes), dtype=np.uint8)
        e2j_matrix = np.ones( (self.num_patch, num_objs+num_objs_across_scenes), dtype=np.uint8)
        for patch_h_i in range(patch_h):
            patch_h_shift = patch_h_i*patch_w
            for patch_w_j in range(patch_w):
                obj_id = obj_2D_patch_anno[patch_h_i, patch_w_j]
                if obj_id != self.undefined and (obj_id in obj_3D_embeddings):
                    obj_idx = obj_3D_id2idx[obj_id]
                    e1i_matrix[patch_h_shift+patch_w_j, obj_idx] = 1 # mark 2D-3D patch-object pairs
                    e2j_matrix[patch_h_shift+patch_w_j, obj_idx] = 0 # mark 2D-3D patch-object unpairs
                
        ## e1j_matrix, (num_patch, num_patch), mark unpaired patch-patch pair for image patches
        e1j_matrix = np.zeros( (self.num_patch, self.num_patch), dtype=np.uint8)
        for patch_h_i in range(patch_h):
            patch_h_shift = patch_h_i*patch_w
            for patch_w_j in range(patch_w):
                obj_id = obj_2D_patch_anno[patch_h_i, patch_w_j]
                if obj_id != self.undefined and obj_id in obj_3D_embeddings:
                    e1j_matrix[patch_h_shift+patch_w_j, :] = np.logical_and(
                        obj_2D_patch_anno_flatten != self.undefined, obj_2D_patch_anno_flatten != obj_id
                    )
                else:
                     e1j_matrix[patch_h_shift+patch_w_j, :] = 1

        ## From 3D to 2D, denote as f1i_matrix, f1j_matrix, f2j_matrix
        ## f1i_matrix = e1i_matrix.T, thus skip
        ## f2j_matrix = e2j_matrix.T, thus skip
        ## f1j_matrix, 1 - I, thus skip
        
        data_dict = {}
        # frame info
        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['image'] = img
        # pano annotations
        # data_dict['patch_anno'] = obj_2D_patch_anno
        # obj info
        data_dict['num_objs'] = len(obj_3D_id2idx)
        data_dict['num_objs_across_scenes'] = num_objs_across_scenes
        data_dict['obj_3D_id2idx'] = obj_3D_id2idx
        data_dict['obj_3D_idx2info'] = obj_3D_idx2info
        data_dict['obj_3D_embeddings_arr'] = obj_3D_embeddings_arr
        # pairs info
        data_dict['e1i_matrix'] = e1i_matrix
        data_dict['e1j_matrix'] = e1j_matrix
        data_dict['e2j_matrix'] = e2j_matrix
        return data_dict
    
    def collate_fn(self, batch):
        batch_size = len(batch)
        
        data_dict = {}
        data_dict['batch_size'] = batch_size
        # frame info 
        data_dict['scan_ids'] = np.stack([data['scan_id'] for data in batch])
        data_dict['frame_idxs'] = np.stack([data['frame_idx'] for data in batch])
        images_batch = np.stack([data['image'] for data in batch])
        data_dict['images'] = torch.from_numpy(images_batch).float() # (B, H, W, C)
        
        # obj info; as obj number is different for each batch, we need to create a list
        data_dict['num_objs'] = [data['num_objs'] for data in batch]
        data_dict['num_objs_across_scenes'] = [data['num_objs_across_scenes'] for data in batch]
        data_dict['obj_3D_id2idx'] = [data['obj_3D_id2idx'] for data in batch]
        data_dict['obj_3D_idx2info'] = \
            [data['obj_3D_idx2info'] for data in batch]
        obj_3D_embeddings_list = [data['obj_3D_embeddings_arr'] for data in batch]
        data_dict['obj_3D_embeddings_list'] = [torch.from_numpy(obj_3D_embeddings).float()
                                               for obj_3D_embeddings in obj_3D_embeddings_list] # B - [N_O, N_Obj_Embed]
        # pairs info
        data_dict['e1i_matrix_list'] = [ torch.from_numpy(data['e1i_matrix']) for data in batch]  # B - [N_P, N_O]
        data_dict['e1j_matrix_list'] = [ torch.from_numpy(data['e1j_matrix']) for data in batch]  # B - [N_P, N_P]
        data_dict['e2j_matrix_list'] = [ torch.from_numpy(data['e2j_matrix']) for data in batch]  # B - [N_P, N_O]
        
        return data_dict
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/room_retrieval_week5/scan3r_cross_scenes.yaml"
    cfg = update_config(config, cfg_file)
    scan3r_ds = PatchObjectPairCrossScenesDataSet(cfg, split='val')
    print(len(scan3r_ds))
    batch = [scan3r_ds[0], scan3r_ds[1]]
    data_batch = scan3r_ds.collate_fn(batch)
    breakpoint=None