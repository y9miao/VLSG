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

class Scan3rOpenSegGenerator():
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
        ## images patch info
        self.image_paths = {}
        if cfg.data.img.rotate:
            image_folder = osp.join(self.scans_files_dir, 'rotate_images')
            for scan_id in self.scan_ids:
                self.image_paths[scan_id] = {}
                frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id)
                for frame_idx in frame_idxs:
                    self.image_paths[scan_id][frame_idx] = osp.join(
                        image_folder, scan_id, "frame-{}.color.jpg".format(frame_idx))
        else:
            for scan_id in self.scan_ids:
                self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id)
            
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
        self.out_dir = osp.join(self.scans_files_dir, out_name)
        common.ensure_dir(self.out_dir)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))
        
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
        img_paths = self.image_paths[scan_id]
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
        # log
        log_str = "Feature generation time: {:.3f}s for {} images, {:.3f}s per image\n".format(
            self.feature_generation_time, img_num, self.feature_generation_time / img_num)
        with open(self.log_file, 'a') as f:
            f.write(log_str)

def main():
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/img_features/OpenSegFeatures/openseg_generator_cfg.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    
    scan3r_openseg_generator = Scan3rOpenSegGenerator(cfg, 'test')
    scan3r_openseg_generator.register_model()
    scan3r_openseg_generator.generateFeatures()
    
if __name__ == "__main__":
    main()