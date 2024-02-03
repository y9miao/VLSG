import argparse
from cgi import test
import os 
import os.path as osp
from re import split
import time
from tracemalloc import start
import comm
from matplotlib import patches
import numpy as np 
import sys
import subprocess

import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ws_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)
# utils
from utils import common
from utils import torch_util
from utils.summary_board import SummaryBoard
# from utils import visualisation
# config
from configs import update_config_room_retrival, config
# tester
from engine.single_tester import SingleTester
from engine import EpochBasedTrainer
from utils.timer import Timer
from utils.common import get_log_string
from utils.summary_board import SummaryBoard
# models
# models
import torch
from torch.nn import functional as F
import torch.optim as optim
## Liploc model
import importlib
from src.models.LipLoc.config.exp_largest_vit import CFG as LipLoc_CFG
import importlib
from src.models.LipLoc.config.exp_largest_vit import CFG as LipLoc_CFG
LipLocModel = importlib.import_module(f"src.models.LipLoc.models.{LipLoc_CFG.model}")
# dataset
from datasets.loaders import get_val_dataloader, get_test_dataloader
from datasets.scan3r_liploc import Scan3rLipLocDataset
# use PathObjAligner for room retrieval
class RoomRetrivalScore():
    def __init__(self, cfg, ds_split = 'val'):
        # cfg
        self.cfg = cfg 
        self.method_name = cfg.val.room_retrieval.method_name
        self.epsilon_th = cfg.val.room_retrieval.epsilon_th
        self.split = ds_split
        
        # get device 
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        # dataloader
        start_time = time.time()
        val_dataset, val_data_loader = get_val_dataloader(cfg, Dataset = Scan3rLipLocDataset)
        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset = Scan3rLipLocDataset)
        # register dataloader
        self.val_data_loader = val_data_loader
        self.val_dataset = val_dataset
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        
        # model
        self.registerModels(cfg)
        self.model.eval()
        self.loss_type = cfg.train.loss.loss_type
        
        # results
        self.val_room_retrieval_summary = SummaryBoard(adaptive=True)
        self.test_room_retrieval_summary = SummaryBoard(adaptive=True)
        
        # files
        self.output_dir = osp.join(cfg.output_dir, self.method_name)
        common.ensure_dir(self.output_dir)
        
        # process time
        self.total_img = 0
        self.img_encoding_time = 0.0
        self.room_retrieval_time = 0.0
        
    def load_snapshot(self, snapshot, fix_prefix=True):
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

        # Load model
        model_dict = state_dict['model']
        self.model.load_state_dict(model_dict, strict=True)
        
    def registerModels(self, cfg):
        # load backbone
        liploc_pretrained_file = cfg.model.backbone3D.pretrained
        self.liploc_model = LipLocModel.Model(LipLoc_CFG)
        self.liploc_model.load_state_dict(torch.load(liploc_pretrained_file), strict=True)

        # model to cuda 
        self.model = self.liploc_model
        if cfg.other.use_resume:
            assert os.path.isfile(cfg.other.resume), "=> no checkpoint found at '{}'".format(cfg.other.resume)
            self.load_snapshot(cfg.other.resume)
        self.model.to(torch.float)
        self.model.to(self.device)
        
        self.test_params = {}
        for name, param in self.model.named_parameters():
            self.test_params[name] = param
            
        # log
        message = 'Model description:\n' + str(self.model)     
 
    def model_forward(self, data_dict):
        # forward
        camera_features = self.model.get_camera_embeddings(data_dict)
        lidar_features = self.model.get_lidar_embeddings(data_dict)
        embeddings = {
            'camera_features': camera_features,
            'lidar_features': lidar_features,
        }
        return embeddings
    def forward_lidar(self, range_imgs):
        range_imgs_features = self.model.encoder_lidar(range_imgs)
        range_imgs_embeddings = self.model.projection_lidar(range_imgs_features)
        return range_imgs_embeddings
    def forward_camera(self, camera_imgs):
        camera_imgs_features = self.model.encoder_camera(camera_imgs)
        camera_imgs_embeddings = self.model.projection_camera(camera_imgs_features)
        return camera_imgs_embeddings

    def room_retrieval_scan(self, data_dict, dataset):
        # room retrieval with scan point cloud
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_S".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_S".format(k): 0. for k in top_k_list}
        retrieval_time_temporal = 0.
        retrieval_time_non_temporal = 0.
        
        for batch_i in range(batch_size):
            cur_img = data_dict['camera_image'][batch_i].unsqueeze(0)
            cur_scan_id = data_dict['scan_ids'][batch_i]
            
            # non-temporal retrieval
            room_score_scans = {}
            scans_embeddings = {}
            ## get cur scan embeddings
            cur_range_img =  dataset.getRangeImagesTensor([cur_scan_id])
            cur_range_img = torch_util.to_cuda(cur_range_img)
            scans_embeddings[cur_scan_id] = self.forward_lidar(cur_range_img).squeeze(0)
            ## get candidate scan embeddings
            candidate_scans = data_dict['candidate_scan_ids_list'][batch_i]

            for candidate_scan_id in candidate_scans:
                range_img = dataset.getRangeImagesTensor([candidate_scan_id])
                range_img = torch_util.to_cuda(range_img)
                scans_embeddings[candidate_scan_id] = self.forward_lidar(range_img).squeeze(0)
            ## calculate similarity
            cur_img_embeddings = self.forward_camera(cur_img).squeeze(0)
            scans_embeddings_cpu = torch_util.release_cuda_torch(scans_embeddings)
            cur_img_embeddings_cpu = torch_util.release_cuda_torch(cur_img_embeddings)
            start_time = time.time()
            for candidate_scan_id, candidate_scan_embeddings in scans_embeddings_cpu.items():
                sim = candidate_scan_embeddings@cur_img_embeddings_cpu.T
                room_score_scans[candidate_scan_id] = sim.max().item()
            candidate_sim_cal_time = time.time() - start_time
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if cur_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_non_temporal["R@{}_NT_S".format(k)] += 1
            retrieval_time_non_temporal += time.time() - start_time
                    
            # temporal retrieval
            temporal_scan_id = data_dict['temporal_scan_id_list'][batch_i]
            ## get temporal depth scan embeddings
            temporal_range_img = dataset.getRangeImagesTensor([temporal_scan_id])
            temporal_range_img = torch_util.to_cuda(temporal_range_img)
            scans_embeddings[temporal_scan_id] = self.forward_lidar(temporal_range_img).squeeze(0)
            ## remove cur scan embeddings
            scans_embeddings.pop(cur_scan_id)
            room_score_scans.pop(cur_scan_id)
            ## calculate similarity
            scans_embeddings_cpu = torch_util.release_cuda_torch(scans_embeddings)
            start_time = time.time()
            sim = scans_embeddings_cpu[temporal_scan_id]@cur_img_embeddings_cpu.T
            room_score_scans[temporal_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if temporal_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            retrieval_time_temporal += time.time() - start_time + candidate_sim_cal_time
            
        # average over batch
        for k in top_k_list:
            top_k_recall_temporal["R@{}_T_S".format(k)] /= 1.0*batch_size
            top_k_recall_non_temporal["R@{}_NT_S".format(k)] /= 1.0*batch_size
        retrieval_time_temporal = retrieval_time_temporal / 1.0*batch_size
        retrieval_time_non_temporal = retrieval_time_non_temporal / 1.0*batch_size
        
        result = {
            'time_T_S': retrieval_time_temporal,
            'time_NT_S': retrieval_time_non_temporal,
        }
        result.update(top_k_recall_temporal)
        result.update(top_k_recall_non_temporal)
        return result

    def room_retrieval_val_test(self):
        # val 
        data_dicts = tqdm.tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader))
        for iteration, data_dict in data_dicts:
            data_dict = torch_util.to_cuda(data_dict)
            result = self.room_retrieval_scan(data_dict, self.val_dataset)
            self.val_room_retrieval_summary.update_from_result_dict(result)
            torch.cuda.empty_cache()
        val_items = self.val_room_retrieval_summary.tostringlist()
        # write to file
        val_file = osp.join(self.output_dir, 'val_result.txt')
        common.write_to_txt(val_file, val_items)
            
        # test
        data_dicts = tqdm.tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader))
        for iteration, data_dict in data_dicts:
            data_dict = torch_util.to_cuda(data_dict)
            result = self.room_retrieval_scan(data_dict, self.test_dataset)
            self.test_room_retrieval_summary.update_from_result_dict(result)
            torch.cuda.empty_cache()
        test_items = self.test_room_retrieval_summary.tostringlist()
        # write to file
        test_file = osp.join(self.output_dir, 'test_result.txt')
        common.write_to_txt(test_file, test_items)
        breakpoint = 1

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    
    cfg = update_config_room_retrival(config, args.config, ensure_dir=True)
    
    # copy config file to out dir
    out_dir = osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name)
    common.ensure_dir(out_dir)
    command = 'cp {} {}'.format(args.config, out_dir)
    subprocess.call(command, shell=True)

    tester = RoomRetrivalScore(cfg)
    tester.room_retrieval_val_test()

if __name__ == '__main__':
    main()