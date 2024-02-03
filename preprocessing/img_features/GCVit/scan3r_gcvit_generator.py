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
import cv2
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
# config
from configs import update_config, config
# GCVit
import torch
import torch.optim as optim
from mmdet.models import build_backbone
from mmcv import Config
from src.models.patch_SG_aligner import PatchSGAligner

class Scan3rGCVitGenerator():
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
        backbone_cfg_file = self.cfg.model.cfg_file
        backbone_cfg = Config.fromfile(backbone_cfg_file)
        backbone_cfg.model['backbone']['pretrained'] = self.cfg.model.pretrained
        backbone = build_backbone(backbone_cfg.model['backbone'])
        self.model = PatchSGAligner(backbone,
                                self.num_reduce,
                                self.backbone_dim,
                                self.img_rotate, 
                                self.patch_hidden_dims,
                                self.patch_encoder_dim,
                                self.obj_embedding_dim,
                                self.obj_embedding_hidden_dims,
                                self.obj_encoder_dim,
                                self.sg_modules,
                                self.sg_rel_dim,
                                self.attr_dim,
                                self.drop)
        self.model.eval()
        self.device = torch.device("cuda")
        self.model.to(self.device)
        
    def inference(self, imgs_tensor):
        feature = self.model.backbone(imgs_tensor)[-1]
        return feature
        
    def generateFeaturesEachScan(self, scan_id):
    # generate VLAD for a scan
    
        # load images
        img_paths = self.image_paths[scan_id]
        img_tensors_dict = {}
        for frame_idx in img_paths:
            img_path = img_paths[frame_idx]
            img  = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (self.image_resize_w, self.image_resize_h),  # type: ignore
                            interpolation=cv2.INTER_LINEAR) # type: ignore
            if self.img_rotate:
                img = img.transpose(1, 0, 2)
                img = np.flip(img, 1)
                img_tensors_dict[frame_idx] = torch.from_numpy(img.copy()).float().to(self.device)   
                
        imgs_features = {}
        
        if False:
            frame_indexs = list(img_paths.keys())
            # aggreate all images to a tensor
            img_tensors_list = []
            for index, frame_idx in enumerate(frame_indexs):
                frame_idx2index = {frame_idx: index }
                img_tensors_list.append(img_tensors_dict[frame_idx])
            imgs_tensor = torch.stack(img_tensors_list, dim=0)
            imgs_tensor_cuda = imgs_tensor
            # inference
            start_time = time.time()
            ## channel first
            imgs_tensor_cuda = imgs_tensor_cuda.permute(0, 3, 1, 2)
            img_features_cuda = self.inference(imgs_tensor_cuda)
            self.feature_generation_time += time.time() - start_time
            ## channel last
            img_features_cuda = img_features_cuda.permute(0, 2, 3, 1)
            img_features = img_features_cuda.cpu().numpy()
            # split features
            for index, frame_idx in enumerate(frame_indexs):
                index = frame_idx2index[frame_idx]
                img_feature = img_features[index]
                imgs_features[frame_idx] = img_feature
        else:
            for frame_idx in img_paths:
                img_tensor = img_tensors_dict[frame_idx]
                img_tensor_cuda = img_tensor
                # inference
                start_time = time.time()
                # channel first 
                img_tensor_cuda = img_tensor_cuda.permute(2, 0, 1)
                img_feature_cuda = self.inference(img_tensor_cuda.unsqueeze(0))
                self.feature_generation_time += time.time() - start_time
                ## channel last
                img_feature_cuda = img_feature_cuda.permute(0, 2, 3, 1)
                img_feature = img_feature_cuda[0].cpu().numpy()
                imgs_features[frame_idx] = img_feature
        return imgs_features
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        for scan_id in tqdm(self.scan_ids):
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

def main():
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/img_features/GCVit/gc_vit_generator.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_gcvit_generator = Scan3rGCVitGenerator(cfg, 'train')
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()