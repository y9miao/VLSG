# get into VLSG space for scan3r data info
import os
import os.path as osp
import sys
from tracemalloc import start
from yaml import scan
vlsg_dir = "/home/yang/big_ssd/Scan3R/VLSG"
sys.path.insert(0, vlsg_dir)
from utils import common, scan3r

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms as tvf
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
from PIL import Image
from tqdm import tqdm

# CLIP
import clip
from clip.model import CLIP

class Scan3rCLIPGenerator():
    def __init__(self, cfg, split):
        self.cfg = cfg
        
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
        
        # CLIP info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
        
        ## out dir 
        self.out_dir = osp.join(self.scans_files_dir, 'Features2D', self.model_name)
        common.ensure_dir(self.out_dir)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))
        
    def register_model(self):
        clip_model, clip_preprocess = clip.load(self.model_version, jit=False)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = torch.device("cuda")
        
    def inference(self, img_path):
        img = Image.open(img_path)
        img_pt = self.clip_preprocess(img).unsqueeze(0).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_pt)
        self.feature_generation_time += time.time() - start_time
            
        return img_features
        
    def generateFeaturesEachScan(self, scan_id):
    # generate VLAD for a scan
    
        # get load images
        img_paths = self.image_paths[scan_id]
        imgs_features = {}
        for frame_idx in img_paths:
            img_path = img_paths[frame_idx]

            # inference
            img_feature = self.inference(img_path)
            
            imgs_features[frame_idx] = img_feature.cpu().numpy().reshape(-1)
            
        return imgs_features
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        for scan_id in tqdm(self.scan_ids):
            imgs_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(imgs_features)
            out_file = osp.join(self.out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(imgs_features, out_file)
        # log
        log_str = "Feature generation time: {:.3f}s for {} images, {:.3f}s per image\n".format(
            self.feature_generation_time, img_num, self.feature_generation_time / img_num)
        with open(self.log_file, 'a') as f:
            f.write(log_str)

def main():
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/img_features/CLIP_features/clip_generator_cfg.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    
    scan3r_anyloc_vlad_generator = Scan3rCLIPGenerator(cfg, 'train')
    scan3r_anyloc_vlad_generator.register_model()
    scan3r_anyloc_vlad_generator.generateFeatures()
    
if __name__ == "__main__":
    main()