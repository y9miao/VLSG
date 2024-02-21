import argparse
from cgi import test
import os 
import os.path as osp
from re import T, split
import time
from tracemalloc import start
import comm
from matplotlib import patches
import numpy as np 
import sys
import subprocess
from collections import OrderedDict
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
# models
# models
import torch
from torch.nn import functional as F
import torch.optim as optim
from models.lidarclip.model.sst import LidarEncoderSST
# from models.GCVit.models import gc_vit
from models.patch_obj_aligner import PatchObjectAligner
from models.loss import ICLLoss
from models.path_obj_pair_visualizer import PatchObjectPairVisualizer
# dataset
from datasets.loaders import get_val_dataloader, get_test_dataloader
from datasets.scan3r_lidarclip import Scan3rLidarClipDataset
from datasets.scannet_lidarclip import ScannetLidarClipDataset
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
        if cfg.data.name == "Scan3R":
            dataset = Scan3rLidarClipDataset
        elif cfg.data.name == "Scannet":
            dataset = ScannetLidarClipDataset
        else:
            raise ValueError("Unknown dataset type: {}".format(cfg.data.name))
        
        start_time = time.time()
        val_dataset, val_data_loader = get_val_dataloader(cfg, Dataset = dataset)
        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset = dataset)
        # register dataloader
        self.val_data_loader = val_data_loader
        self.val_dataset = val_dataset
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        self.val_scan_pcs = self.val_dataset.load_scan_pcs()
        self.test_scan_pcs = self.test_dataset.load_scan_pcs()
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        
        # model
        self.registerModels(cfg)
        self.model.eval()
        self.loss_type = cfg.train.loss.loss_type
        
        # results
        self.val_room_retrieval_summary = SummaryBoard(adaptive = True)
        self.test_room_retrieval_summary = SummaryBoard(adaptive = True)
        
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
        backbone3d_cfg_file = cfg.model.backbone3D.cfg_file
        backbone3d_pretrained_file = cfg.model.backbone3D.pretrained
        dim_3d = cfg.model.backbone3D.dim_3d
        self.backbone3d = LidarEncoderSST(backbone3d_cfg_file, dim_3d)
        self.backbone3d.load_state_dict(torch.load(backbone3d_pretrained_file), strict=False)

        # model to cuda 
        self.model = self.backbone3d
        if cfg.other.use_resume:
            assert os.path.isfile(cfg.other.resume), "=> no checkpoint found at '{}'".format(cfg.other.resume)
            self.load_snapshot(cfg.other.resume)
        self.model.to(torch.float)
        self.model.to(self.device)
 
    def model_forward(self, pcs):
        embeddings_3D_pcs = self.model(pcs)
        return embeddings_3D_pcs
 
    def sim_calculation_eval(self, embeddings_2D, embeddings_3D):
        if self.loss_type == 'mse':
            sim_mse = -torch.norm(embeddings_2D - embeddings_3D, dim=1) # l2 sim
            return sim_mse, None
        else:
            embeddings_3D_norm = F.normalize(embeddings_3D, dim=1)
            embeddings_2D_norm = F.normalize(embeddings_2D, dim=1)
            sim_cos = torch.mm(embeddings_2D_norm, embeddings_3D_norm.permute(1, 0)) # cos sim
            return None, sim_cos

    def room_retrieval_scan(self, data_dict, scan_pcs):
        # room retrieval with scan point cloud
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_S".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_S".format(k): 0. for k in top_k_list}
        retrieval_time_temporal = 0.
        retrieval_time_non_temporal = 0.
        
        for batch_i in range(batch_size):
            img_feature = data_dict['img_features'][batch_i].unsqueeze(0)
            cur_scan_id = data_dict['scan_ids'][batch_i]
            
            # non-temporal retrieval
            room_score_scans = {}
            scans_pcs_embeddings = {}
            ## get cur scan embeddings
            # curr_scan_pcs = data_dict['curr_scan_pcs_list'][batch_i]
            curr_scan_pcs = torch_util.to_cuda(scan_pcs[cur_scan_id])
            pcs = [curr_scan_pcs.contiguous()]
            scans_pcs_embeddings[cur_scan_id], _ = self.model_forward(pcs)
            ## get candidate scan embeddings
            # candidate_scans_pcs = data_dict['candidate_scan_pcs_list'][batch_i]
            candidate_scans = data_dict['candidate_scan_ids_list'][batch_i]
            candidate_scans_pcs = {scan_id: torch_util.to_cuda(scan_pcs[scan_id]) for scan_id in candidate_scans}
            for candidate_scan_id, scans_pcs in candidate_scans_pcs.items():
                pcs = [scans_pcs.contiguous()]
                scans_pcs_embeddings[candidate_scan_id], _ = self.model_forward(pcs)
            ## calculate similarity in cpu
            img_feature_cpu = img_feature.cpu()
            scans_pcs_embeddings_cpu = torch_util.release_cuda_torch(scans_pcs_embeddings)
            start_time = time.time()
            for candidate_scan_id, candidate_scan_embeddings in scans_pcs_embeddings_cpu.items():
                sim_mse, sim_cos = self.sim_calculation_eval(img_feature_cpu, candidate_scan_embeddings)
                sim = sim_mse if self.loss_type == 'mse' else sim_cos
                room_score_scans[candidate_scan_id] = sim.max().item()
            candidate_sim_cal_time = time.time() - start_time
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if cur_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_non_temporal["R@{}_NT_S".format(k)] += 1
            # record time
            retrieval_time_non_temporal += time.time() - start_time
                    
            # temporal retrieval
            temporal_scan_id = data_dict['temporal_scan_id_list'][batch_i]
            ## get temporal depth scan embeddings
            # temporal_scan_pcs = data_dict['temporal_scan_pcs_list'][batch_i]
            temporal_scan_pcs = torch_util.to_cuda(scan_pcs[temporal_scan_id])
            pcs = [temporal_scan_pcs.contiguous()]
            scans_pcs_embeddings[temporal_scan_id],_ = self.model_forward(pcs)
            ## remove cur scan embeddings
            if cur_scan_id != temporal_scan_id:
                scans_pcs_embeddings.pop(cur_scan_id)
                room_score_scans.pop(cur_scan_id)
            ## calculate 
            scans_pcs_embeddings_cpu = torch_util.release_cuda_torch(scans_pcs_embeddings)
            start_time = time.time()
            sim_mse, sim_cos = self.sim_calculation_eval(img_feature_cpu, scans_pcs_embeddings_cpu[temporal_scan_id])
            sim = sim_mse if self.loss_type == 'mse' else sim_cos
            room_score_scans[temporal_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if temporal_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            # record time
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
        # test
        data_dicts = tqdm.tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader))
        for iteration, data_dict in data_dicts:
            with torch.no_grad():
                data_dict = torch_util.to_cuda(data_dict)
                result = self.room_retrieval_scan(data_dict, self.test_scan_pcs)
                self.test_room_retrieval_summary.update_from_result_dict(result)
                torch.cuda.empty_cache()
        test_items = self.test_room_retrieval_summary.tostringlist()
        # write to file
        test_file = osp.join(self.output_dir, 'test_result.txt')
        common.write_to_txt(test_file, test_items)
        
        # val 
        data_dicts = tqdm.tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader))
        for iteration, data_dict in data_dicts:
            with torch.no_grad():
                data_dict = torch_util.to_cuda(data_dict)
                result = self.room_retrieval_scan(data_dict, self.val_scan_pcs)
                self.val_room_retrieval_summary.update_from_result_dict(result)
                torch.cuda.empty_cache()
            
        val_items = self.val_room_retrieval_summary.tostringlist()
        # write to file
        val_file = osp.join(self.output_dir, 'val_result.txt')
        common.write_to_txt(val_file, val_items)
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