# get into VLSG space for scan3r data info
import os
import os.path as osp
import sys
from tracemalloc import start
from sklearn.utils import resample
from yaml import scan
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common, scan3r

import numpy as np
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms as tvf
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tyro
import time
import traceback
import joblib
import wandb
import argparse
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal, Tuple, List, Union
from dataclasses import dataclass, field
# config
from configs import update_config, config
# Dino v2
from dinov2_utils import DinoV2ExtractFeatures
from dataclasses import dataclass
# @dataclass
# class LocalArgs:
#     """
#         Local arguments for the program
#     """
#     # Dino_v2 properties (parameters)
#     desc_layer: int = 31
#     desc_facet: Literal["query", "key", "value", "token"] = "value"
#     num_c: int = 32
#     # device
#     device = torch.device("cuda")
# larg = tyro.cli(LocalArgs)

class Scan3rDinov2Generator():
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
        self.inference_step = cfg.data.inference_step
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
        
        # model info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
        ## 2Dbackbone
        self.num_reduce = cfg.model.backbone.num_reduce
        self.backbone_dim = cfg.model.backbone.backbone_dim
        self.img_rotate = cfg.data.img_encoding.img_rotate
        ## scene graph encoder
        self.sg_modules = cfg.sgaligner.modules
        self.sg_rel_dim = cfg.sgaligner.model.rel_dim
        self.attr_dim = cfg.sgaligner.model.attr_dim
        ## encoders
        self.patch_hidden_dims = cfg.model.patch.hidden_dims
        self.patch_encoder_dim = cfg.model.patch.encoder_dim
        self.obj_embedding_dim = cfg.model.obj.embedding_dim
        self.obj_embedding_hidden_dims = cfg.model.obj.embedding_hidden_dims
        self.obj_encoder_dim = cfg.model.obj.encoder_dim
        self.drop = cfg.model.other.drop
        
        # image preprocessing 
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.step = self.cfg.data.img.img_step
        
        ## out dir 
        self.out_dir = osp.join(self.scans_files_dir, 'Features2D', self.model_name)
        common.ensure_dir(self.out_dir)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))
        
    def register_model(self):
        
        desc_layer = 31
        desc_facet = "value"
        device = torch.device("cuda")
        self.device = torch.device("cuda")
        
        # Dinov2 extractor
        if "extractor" in globals():
            print(f"Extractor already defined, skipping")
        else:
            self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
                desc_facet, device=device)

        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
    def inference(self, imgs_tensor):
        feature = self.model.backbone(imgs_tensor)[-1]
        return feature
        
    def generateFeaturesEachScan(self, scan_id):
    # generate VLAD for a scan
        imgs_features = {}
        # load images
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        
        for infer_step_i in range(0, len(frame_idxs_list)//self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

            tensor_idxs_to_frame_idxs = {}
            img_tensors_list = []
            if len(frame_idxs_sublist) == 0:
                continue
            
            for idx, frame_idx in enumerate(frame_idxs_sublist):
                img_path = img_paths[frame_idx]
                img  = Image.open(img_path).convert('RGB')
                h_new, w_new = (self.image_resize_h // 14) * 14, (self.image_resize_w // 14) * 14
                img_pt = img.resize((w_new, h_new), Image.BICUBIC)
                if self.img_rotate:
                    img_pt = img_pt.transpose(Image.ROTATE_270)

                img_pt = self.base_tf(img_pt)
                img_tensors_list.append(img_pt)
                tensor_idxs_to_frame_idxs[idx] = frame_idx
                
            imgs_tensor = torch.stack(img_tensors_list, dim=0).float().to(self.device)
            # inference
            start_time = time.time()
            ret = self.extractor(imgs_tensor) # [num_images, num_patches, desc_dim]  
            self.feature_generation_time += time.time() - start_time
            
            
            for idx, frame_idx in tensor_idxs_to_frame_idxs.items():
                imgs_features[frame_idx] = ret[idx].cpu().numpy()
                
        return imgs_features
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        for scan_id in tqdm(self.scan_ids[3:]):
            with torch.no_grad():
                imgs_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(imgs_features)
            out_file = osp.join(self.out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(imgs_features, out_file)
        # log
        log_str = "Feature generation time: {:.3f}s for {} images, {:.3f}s per image\n".format(
            self.feature_generation_time, img_num, self.feature_generation_time / img_num)
        with open(self.log_file, 'a') as f:
            f.write(log_str)

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    return parser.parse_known_args()

def main():

    # get arguments
    args, _ = parse_args()
    cfg_file = args.config

    from configs import config, update_config
    cfg = update_config(config, cfg_file, ensure_dir = False)
    
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train')
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val')
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test')
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()