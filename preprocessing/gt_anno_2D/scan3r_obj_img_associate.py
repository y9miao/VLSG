import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
import argparse
import cv2
from tqdm import tqdm
import sys
ws_dir = "/home/yang/big_ssd/Scan3R/VLSG"
sys.path.insert(0, ws_dir)

from configs import config, update_config
from utils import common, scan3r

# associate image patch and obj id

class Scan3ROBJAssociator():
    def __init__(self, data_root_dir, split, cfg):
        self.cfg = cfg
        self.split = split
        self.use_rescan = self.cfg.data.rescan
        self.data_root_dir = data_root_dir
        
        scan_dirname = ''
        self.scans_dir = osp.join(data_root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        self.scenes_config_file = osp.join(self.scans_dir, 'files', '3RScan.json')
        self.scenes_configs = common.load_json(self.scenes_config_file)
        self.objs_config_file = osp.join(self.scans_dir, 'files', 'objects.json')
        self.objs_configs = common.load_json(self.objs_config_file)
        self.scan_ids = []
        
        self.step = self.cfg.data.img.img_step
        
        # get scans
        for scan_data in self.scenes_configs:
            if scan_data['type'] == self.split:
                self.scan_ids.append(scan_data['reference'])
                if self.use_rescan:
                    rescan_ids = [scan['reference'] for scan in scan_data['scans']]
                    self.scan_ids += rescan_ids
                    
        self.scan_ids.sort()
        
        # 2D object id annotation
        self.obj_2D_anno_dir = osp.join(self.scans_dir, 'files', 'gt_projection', 'obj_id')
        
        # 2D image pObjectEmbeddingGeneratoratch annotation
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.patch_w_size = self.image_w / self.image_patch_w
        self.patch_h_size = self.image_h / self.image_patch_h
        self.patch_anno_folder_name = "patch_anno_{}_{}".format(self.image_patch_w, self.image_patch_h)
        self.anno_out_dir = osp.join(self.scans_dir, 'files', 'patch_anno', self.patch_anno_folder_name)
        common.ensure_dir(self.anno_out_dir)
        
      
    def __len__(self):
        return len(self.anchor_data)

    def annotate(self, scan_idx, step, th=0.2):
        # get related files
        scan_id = self.scan_ids[scan_idx]
        # get frame annotations
        frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id, step)
        gt_2D_obj_anno_imgs = scan3r.load_gt_2D_anno(self.data_root_dir, scan_id, step)
        
        patch_annos_scan = {}
        # iterate over images
        for frame_idx in frame_idxs:
            gt_2D_obj_anno_img = gt_2D_obj_anno_imgs[frame_idx]
            patch_annos = np.zeros((self.image_patch_h, self.image_patch_w), dtype=np.uint8)
            # iterate over patches within one image
            for patch_h_i in range(self.image_patch_h):
                h_start = round(patch_h_i * self.patch_h_size)
                h_end = round((patch_h_i+1) * self.patch_h_size)
                for patch_w_j in range(self.image_patch_w):
                    w_start = round(patch_w_j * self.patch_w_size)
                    w_end = round((patch_w_j+1) * self.patch_w_size)
                    patch_size = (w_end - w_start) * (h_end - h_start)
                    patch_anno = gt_2D_obj_anno_img[h_start:h_end, w_start:w_end]
                    obj_ids, counts = np.unique(patch_anno.reshape(-1), return_counts=True)
                    max_idx = np.argmax(counts)
                    max_count = counts[max_idx]
                    if(max_count > th*patch_size):
                        patch_annos[patch_h_i,patch_w_j] = obj_ids[max_idx]
            
            patch_annos_scan[frame_idx] = patch_annos   
                    
        return patch_annos_scan
    
    def annotate_scans(self):
        self.patch_annos_scans = {}
        for scan_idx in tqdm(range(len(self.scan_ids))):
            patch_annos_scan = self.annotate(scan_idx, self.step)
            self.patch_annos_scans[scan_idx] = patch_annos_scan
            
        # save file
        for scan_idx in tqdm(range(len(self.scan_ids))):
            scan_id = self.scan_ids[scan_idx]
            patch_anno_file = osp.join(self.anno_out_dir, scan_id+".pkl")
            common.write_pkl_data(self.patch_annos_scans[scan_idx], patch_anno_file)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    return parser.parse_known_args()
        
if __name__ == '__main__':
    # get arguments
    args, _ = parse_args()
    cfg_file = args.config
    
    # get env variable for Data_ROOT_DIR
    Data_ROOT_DIR = os.getenv('Data_ROOT_DIR')
    
    cfg = update_config(config, cfg_file, ensure_dir = False)
    split = "validation"
    scan3r_img_projector = Scan3ROBJAssociator(Data_ROOT_DIR, split=split, cfg=cfg)
    scan3r_img_projector.annotate_scans()
    split = "train"
    scan3r_img_projector = Scan3ROBJAssociator(Data_ROOT_DIR, split=split, cfg=cfg)
    scan3r_img_projector.annotate_scans()