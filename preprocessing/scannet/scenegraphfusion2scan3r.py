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

import subprocess

class ScannetSGPrediction():
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
                
        # prediction folder
        self.pred_folders = {}
        for scan_id in self.scan_ids:
            scan_folder = osp.join(cfg.data.root_dir, split, scan_id)
            self.pred_folders[scan_id] = osp.join(scan_folder, "scene_graph_fusion")
               
        # some cfgs
        rel_class_file = osp.join(cfg.data.root_dir, 'files', 'scannet8_relationships.txt')
        self.rel2idx = common.name2idx(rel_class_file)
        obj_class_file = osp.join(cfg.data.root_dir, 'files', 'scannet20_classes.txt')
        self.class2idx = common.name2idx(obj_class_file)
            
        # out dir
        self.out_dir = osp.join(cfg.data.root_dir, split)
        
    def SceneGraphPrediction2Scan3r(self):
        for scan_id in tqdm(self.scan_ids):
            pred_folder = self.pred_folders[scan_id]
            # data_dict = scannet_utils.scenegraphfusion2scan3r(scan_id, pred_folder, 
            #                 self.rel2idx, self.class2idx, self.cfg)
            # save data
            file = osp.join(pred_folder, "{}.pkl".format(scan_id))
            # common.write_pkl_data(data_dict, file)
            
            # get edge features
            scannet_utils.calculate_bow_node_edge_feats(file, self.rel2idx)
    
def main():
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/scannet/scannet_sg2scan3r.yaml"
    cfg = CN()
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    
    scannet_dino_generator = ScannetSGPrediction(cfg, split='val')
    scannet_dino_generator.SceneGraphPrediction2Scan3r()
    
if __name__ == '__main__':
    main()