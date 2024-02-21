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

cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/scannet/scannet_sg_prediction.yaml"
scene_graph_fusion_exe = "/home/yang/toolbox/third_party/SceneGraphFusion/bin/exe_GraphSLAM"
scene_graph_fusion_model = "/home/yang/toolbox/third_party/SceneGraphFusion/traced/"
parallel_jobs = 4

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
        # out dir
        self.out_dir = osp.join(cfg.data.root_dir, split)
        
    def generateSceneGraphPrediction(self):
        commands = []
        for scan_id in tqdm(self.scan_ids):
            scan_folder = osp.join(self.out_dir, scan_id)
            seq_file = osp.join(scan_folder, "{}.sens".format(scan_id))
            out_folder = osp.join(scan_folder, "scene_graph_fusion")
            
            exe_command = "{} --pth_in {} --pth_out {} --pth_model {}".format(
                scene_graph_fusion_exe, seq_file, out_folder, scene_graph_fusion_model)
            commands.append(exe_command)
        # run the commands
        scannet_utils.RunBashBatch(commands, jobs_per_step=parallel_jobs)
            
        all_processed = False
        while not all_processed:
            
            commands = []
            # check if the output file exists
            ## if not exists, add to the commands
            for scan_id in self.scan_ids:
                out_folder = osp.join(self.out_dir, scan_id, "scene_graph_fusion")
                files_to_check = ['predictions.json', 'inseg.ply', "node_semantic.ply"]
                
                if not all([osp.exists(osp.join(out_folder, file)) for file in files_to_check]):
                    scan_folder = osp.join(self.out_dir, scan_id)
                    seq_file = osp.join(scan_folder, "{}.sens".format(scan_id))
                    exe_command = "{} --pth_in {} --pth_out {} --pth_model {}".format(
                        scene_graph_fusion_exe, seq_file, out_folder, scene_graph_fusion_model)
                    commands.append(exe_command)
            all_processed = len(commands) == 0
            # run the commands
            scannet_utils.RunBashBatch(commands, jobs_per_step=parallel_jobs)
            
    
def main():
    cfg = CN()
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    
    scannet_dino_generator = ScannetSGPrediction(cfg, split='test')
    scannet_dino_generator.generateSceneGraphPrediction()
    
if __name__ == '__main__':
    main()