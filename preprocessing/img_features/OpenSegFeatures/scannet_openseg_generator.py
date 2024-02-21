# get into VLSG space for scan3r data info
import os
import os.path as osp
import sys
from tracemalloc import start
from yaml import scan
vlsg_dir = "/home/yang/big_ssd/Scan3R/VLSG"
sys.path.insert(0, vlsg_dir)
from utils import common, scan3r, scannet_utils

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
from tensorflow import io
import tensorflow.compat.v1 as tf
import tensorflow as tf2

def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''
    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes

# CLIP
import clip
from clip.model import CLIP

class ScannetOpenSegGenerator():
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
                
        # get image paths
        self.img_step = cfg.data.img_step
        self.data_split_dir = osp.join(cfg.data.root_dir, split)
        self.img_paths = {}
        for scan_id in self.scan_ids:
            img_paths = scannet_utils.load_frame_paths(self.data_split_dir, scan_id, self.img_step)
            self.img_paths[scan_id] = img_paths
        
        # image patch 
        self.image_w = cfg.data.img.w
        self.image_h = cfg.data.img.h
        self.image_patch_w = cfg.data.img_encoding.patch_w
        self.image_patch_h = cfg.data.img_encoding.patch_h
        self.patch_w_size = self.image_w // self.image_patch_w
        self.patch_h_size = self.image_h // self.image_patch_h
        
        # Openseg info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
        self.model_path = cfg.model.model_path
        self.feat_dim = cfg.data.img_encoding.feat_dim
        self.text_emb = tf.zeros([1, 1, self.feat_dim])
        
        ## out dir 
        out_name = cfg.data.img_encoding.feature_dir
        self.scans_files_dir = osp.join(cfg.data.root_dir, 'files')
        self.out_dir = osp.join(self.scans_files_dir, out_name)
        common.ensure_dir(self.out_dir)
                
    def register_model(self):
        self.openseg_model = tf2.saved_model.load(self.model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        
    def inference(self, img_path):
        np_image_string = read_bytes(img_path)

        start_time = time.time()
        results = self.openseg_model.signatures['serving_default'](
                inp_image_bytes=tf.convert_to_tensor(np_image_string),
                inp_text_emb = self.text_emb)
        image_embedding_feat = results['ppixel_ave_feat'] # 1x640x640x768
        self.feature_generation_time += time.time() - start_time
    
        img_info = results['image_info']
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1])
        ]
        patch_h_size = crop_sz[0] // self.image_patch_h
        patch_w_size = crop_sz[1] // self.image_patch_w
        
        image_embedding = image_embedding_feat[0, :crop_sz[0], :crop_sz[1]].numpy() # 640x360x768
        patches_embedding = np.zeros((self.image_patch_h, self.image_patch_w, self.feat_dim))
        for h_i in range(self.image_patch_h):
            for w_i in range(self.image_patch_w):
                # take average of openseg features of each patch 
                patch_embedding = image_embedding[h_i*patch_h_size:(h_i+1)*patch_h_size, 
                                w_i*patch_w_size:(w_i+1)*patch_w_size].mean(axis=(0, 1))
                patches_embedding[h_i, w_i] = patch_embedding
        return patches_embedding
        
    def generateFeaturesEachScan(self, scan_id):
    # generate VLAD for a scan
    
        # get load images
        img_paths = self.img_paths[scan_id]
        imgs_features = {}
        for frame_idx in img_paths:
            img_path = img_paths[frame_idx]

            # inference
            img_feature = self.inference(img_path)
            
            imgs_features[frame_idx] = img_feature
            
        return imgs_features
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        for scan_id in tqdm(self.scan_ids):
            imgs_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(imgs_features)
            out_file = osp.join(self.out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(imgs_features, out_file)

def main():
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/990Pro/scannet_seqs/data"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/img_features/OpenSegFeatures/openseg_generator_scannet_cfg.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    
    scan3r_openseg_generator = ScannetOpenSegGenerator(cfg, 'test')
    scan3r_openseg_generator.register_model()
    scan3r_openseg_generator.generateFeatures()
    
if __name__ == "__main__":
    main()