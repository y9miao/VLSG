import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
import cv2
import sys
sys.path.append('..')
sys.path.append('../..')
from utils import common, scan3r

class PatchObjectPairDataSet(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
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
        
        # scans
        self.scan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_scans.txt'.format(split)), dtype=str)
        self.rescan = cfg.data.rescan
        rescan_note = ''
        if self.rescan:
            rescan_note = 're'
            rescan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, rescan_note)), dtype=str)
            self.scan_ids = np.concatenate([self.scan_ids, rescan_ids])
            
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
            
        # load 2D patch annotation
        self.obj_2D_patch_anno = {}
        for scan_id in self.scan_ids:
            patch_anno_scan_file = osp.join(self.scans_2Dpatch_anno_dir, "{}.pkl".format(scan_id))
            self.obj_2D_patch_anno[scan_id] = common.load_pkl_data(patch_anno_scan_file)
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()
            
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
                
        return data_items
    
    def __getitem__(self, idx):
        data_item = self.data_items[idx]

        # 3D object embeddings
        scan_id = data_item['scan_id']
        obj_3D_embeddings = self.obj_3D_embeddings[scan_id]
        obj_3D_embeddings_arr = np.array([obj_3D_embeddings[obj_id] for obj_id in obj_3D_embeddings])
        num_objs = len(obj_3D_embeddings)
        obj_3D_id2idx = {}
        idx = 0
        for obj_id in obj_3D_embeddings:
            obj_3D_id2idx[obj_id] = idx
            idx += 1
            
        # img data
        img_path = data_item['img_path']
        frame_idx = data_item['frame_idx']
        img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.image_resize_w, self.image_resize_h), 
                         interpolation=cv2.INTER_LINEAR)
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
        ## e1i_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object pairs
        ## e2j_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object unpairs
        e1i_matrix = np.zeros( (self.num_patch, num_objs), dtype=np.uint8)
        e2j_matrix = np.ones( (self.num_patch, num_objs), dtype=np.uint8)
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

        ## e2i to be explored TODO
        # firstly only try one side(patch to object) ...
        
        data_dict = {}
        # frame info
        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['image'] = img
        # pano annotations
        data_dict['patch_anno'] = obj_2D_patch_anno
        # obj info
        data_dict['obj_3D_id2idx'] = obj_3D_id2idx
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
        
        # data_dict['obj_3D_id2idx'] = np.stack([data['obj_3D_id2idx'] for data in batch])
        # obj_3D_embeddings_arrs = np.stack([data['obj_3D_embeddings_arr'] for data in batch])
        # data_dict['obj_3D_embeddings_arr'] = torch.from_numpy(
        #     obj_3D_embeddings_arrs).type(torch.FloatTensor) # (B, N_O, N_Obj_Embed)
        
        # obj info; as obj number is different for each batch, we need to create a list
        data_dict['obj_3D_id2idx'] = [data['obj_3D_id2idx'] for data in batch]
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
    cfg_file = "/home/yang/toolbox/ECCV2024/CodePlace/VLSG/configs/obj_match_scan3r.yaml"
    cfg = update_config(config, cfg_file)
    scan3r_ds = PatchObjectPairDataSet(cfg, split='val')
    print(len(scan3r_ds))
    batch = [scan3r_ds[1800], scan3r_ds[1801]]
    data_batch = scan3r_ds.collate_fn(batch)
    breakpoint=None