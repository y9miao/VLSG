import os
import os.path as osp
from re import I
import sys
from tracemalloc import start
from yacs.config import CfgNode as CN
import comm
from sklearn.utils import resample
from yaml import scan
vlsg_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, vlsg_dir)
from utils import common, scannet_utils

import numpy as np
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms as tvf
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal, Tuple, List, Union
from dataclasses import dataclass, field

# CLIP
import clip
from clip.model import CLIP

class ScannetClipGenerator():
    def __init__(self, cfg, split):
        self.cfg = cfg
        
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
        # out dir
        feature2D_name = cfg.data.feature_2D_name
        self.feat_2D_out_dir = osp.join(cfg.data.root_dir, "files", feature2D_name)
        common.ensure_dir(self.feat_2D_out_dir)
        
        # get image paths
        self.img_step = cfg.data.img_step
        self.data_split_dir = osp.join(cfg.data.root_dir, split)
        self.img_paths = {}
        for scan_id in self.scan_ids:
            img_paths = scannet_utils.load_frame_paths(self.data_split_dir, scan_id, self.img_step)
            self.img_paths[scan_id] = img_paths
            
        # feature inference config
        self.inference_step = cfg.data.inference_step
        self.image_resize_w = self.cfg.data.resize_w
        self.image_resize_h = self.cfg.data.resize_h
        
        # CLIP info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
                
    def register_model(self):
        
        clip_model, clip_preprocess = clip.load(self.model_version, jit=False)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = torch.device("cuda")
        
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                imgs_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(imgs_features)
            # out_file = osp.join(self.feat_2D_out_dir, '{}.pkl'.format(scan_id))
            # common.write_pkl_data(imgs_features, out_file)
            
            # save features in frame-level
            out_scan_folder = osp.join(self.feat_2D_out_dir, scan_id)
            common.ensure_dir(out_scan_folder)
            for frame_idx in imgs_features:
                out_file = osp.join(out_scan_folder, '{}.npy'.format(frame_idx))
                np.save(out_file, imgs_features[frame_idx])
        
    def generateFeaturesEachScan(self, scan_id):
        # generate DINOV2 features for a scan
        imgs_features = {}
        # load images
        img_paths = self.img_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        
        for infer_step_i in range(0, len(frame_idxs_list)//self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]
            
            tensor_idxs_to_frame_idxs = {}
            img_pt_list = []
            if len(frame_idxs_sublist) == 0:
                continue
            
            for idx, frame_idx in enumerate(frame_idxs_sublist):
                img_path = img_paths[frame_idx]
                img  = Image.open(img_path).convert('RGB')
                img_pt = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                img_pt_list.append(img_pt)
                tensor_idxs_to_frame_idxs[idx] = frame_idx
            imgs_pt_tensor = torch.cat(img_pt_list, dim=0)  
             # inference
            with torch.no_grad():
                img_features = self.clip_model.encode_image(imgs_pt_tensor)  
            
            for idx, frame_idx in tensor_idxs_to_frame_idxs.items():
                imgs_features[frame_idx] = img_features[idx].cpu().numpy()
        return imgs_features
    
def main():
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/scannet/scannet_clip_features.yaml"
    cfg = CN()
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    
    scannet_dino_generator = ScannetClipGenerator(cfg, split='test')
    scannet_dino_generator.register_model()
    scannet_dino_generator.generateFeatures()
    
if __name__ == '__main__':
    main()