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
import open3d as o3d
import subprocess
from concurrent.futures import ThreadPoolExecutor

class ScannetGT2Scan3r():
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
        self.scene_folder = {}
        for scan_id in self.scan_ids:
            scene_folder = osp.join(cfg.data.root_dir, split, scan_id)
            self.scene_folder[scan_id] = scene_folder
               
        # some cfgs
        label_map_file = osp.join(cfg.data.root_dir, 'files', 'scannetv2-labels.combined.tsv')
        self.label_map = scannet_utils.read_label_mapping(label_map_file)
            
        # out dir
        self.out_dir = osp.join(cfg.data.root_dir, split)
        
    def SceneGraphGt2Scan3r(self, num_worker = 1, vis=False):
        arguments = []
        for scan_id in tqdm(self.scan_ids):
            scene_folder = self.scene_folder[scan_id]
            arguments.append((scan_id, scene_folder))
        
        # parallelly process each scan with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(tqdm(executor.map(self.SceneGraphGt2Scan3rEachScan, 
                                             [arg[0] for arg in arguments],
                                             [arg[1] for arg in arguments]), 
                                total=len(arguments), leave=False))
            
    def SceneGraphGt2Scan3rEachScan(self, scan_id, scene_folder):
        scene_folder = self.scene_folder[scan_id]
        data_dict = scannet_utils.scannetGtSeg2scan3r(scan_id, scene_folder, 
                        self.label_map, self.cfg)
        # save data
        out_folder = osp.join(scene_folder, 'gt_scan3r')
        file = osp.join(out_folder, "{}.pkl".format(scan_id))
        common.write_pkl_data(data_dict, file)
    
def main():
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/scannet/scannet_gt2scan3r.yaml"
    cfg = CN()
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    
    scannet_dino_generator = ScannetGT2Scan3r(cfg, split='test')
    scannet_dino_generator.SceneGraphGt2Scan3r(num_worker=7)
    
if __name__ == '__main__':
    main()