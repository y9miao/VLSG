# get into VLSG space for scannet data info
from ast import arg
import os
import os.path as osp
import sys
from tracemalloc import start
from yaml import scan

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
from zmq import device

class ScannetEvaGenerator():
    def __init__(self, cfg, split, device, uni3d_dir, log_dir):
        import common, scannet_utils
        self.cfg = cfg
        self.device = torch.device(device)
        
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
        
        # get image paths of fall images for each scan, step will be down externally
        self.data_split_dir = osp.join(cfg.data.root_dir, split)
        self.img_paths = {}
        for scan_id in self.scan_ids:
            scan_folder = osp.join(self.data_split_dir, scan_id)
            image_folder = osp.join(scan_folder, 'color')
            img_names = os.listdir(image_folder)
            self.img_paths[scan_id] = {img_name: osp.join(image_folder, img_name) for img_name in img_names}
        
        self.log_file = osp.join(log_dir, "log_file_scannet_{}.txt".format(self.split))
        
    def register_model(self, ckpt):
        # eva
        import open_clip
        clip_model_name = "EVA02-E-14-plus" 
        self.clip_model, _ , _ = open_clip.create_model_and_transforms(
            model_name = clip_model_name, pretrained=ckpt)
        self.clip_model.eval()
        self.clip_model.to(self.device)
        
    def inference(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        # resize 
        img = img.resize((224, 224))
        # to tensor
        img_pt = T.ToTensor()(img).to(self.device).unsqueeze(0)
        # # already channel first
        # img_pt = img_pt.permute(0, 3, 1, 2)

        start_time = time.time()
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_pt)
        self.feature_generation_time += time.time() - start_time
            
        return img_features
        
    def generateFeaturesEachScan(self, scan_id):
    # generate VLAD for a scan
    
        # get load images
        img_paths = self.img_paths[scan_id]
        imgs_features = {}
        for frame_idx in img_paths:
            img_path = img_paths[frame_idx]

            # inference
            img_feature = self.inference(img_path)
            
            imgs_features[frame_idx] = img_feature.cpu().numpy().reshape(-1)
            
        return imgs_features
    
    def generateFeatures(self):
        import common
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
            
# args
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Scannet 3D feature generator')
    parser.add_argument('--cfg', type=str, default='', help='path to config file')
    parser.add_argument('--split', type=str, default='test', help='split')
    # device
    parser.add_argument('--device', type=str, default='cuda', help='device')
    # add args of root dir
    parser.add_argument('--root_dir', type=str, default='', help='root dir')
    # uni3D path
    parser.add_argument('--uni3d_dir', type=str, default='', help='uni3D dir')
    # vlsg path
    parser.add_argument('--vlsg_dir', type=str, default='', help='vlsg dir')
    # ckpt
    parser.add_argument('--ckpt', type=str, default='', help='ckpt')
    # log dir
    parser.add_argument('--log_dir', type=str, default='', help='log dir')
        
    args = parser.parse_args()
    return args

def main(args):
    args = parse_args()
    root_dir = args.root_dir
    cfg_file = args.cfg
    split = args.split
    device = args.device
    # get into Uni3D space
    uni3d_dir = args.uni3d_dir
    # vlsg space
    vlsg_dir = args.vlsg_dir
    # ckpt
    ckpt = args.ckpt
    # log dir
    log_dir = args.log_dir
    
    # vlsd space
    # sys.path.append(vlsg_dir)
    utils_dir = osp.join(vlsg_dir, 'utils')
    sys.path.insert(0, vlsg_dir)
    sys.path.insert(0, utils_dir)
    import common, scan3r
    # cfg
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = root_dir
    cfg_file = cfg_file
    cfg = update_config(config, cfg_file, ensure_dir = False)
    
    scan3r_anyloc_vlad_generator = ScannetEvaGenerator(cfg, split, device, uni3d_dir, log_dir)
    scan3r_anyloc_vlad_generator.register_model(ckpt)
    scan3r_anyloc_vlad_generator.generateFeatures()
    
if __name__ == "__main__":
    main(sys.argv[1:])