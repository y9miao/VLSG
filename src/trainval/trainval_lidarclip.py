import argparse
from math import isnan
import time
import os
from collections import OrderedDict
import sys
import numpy as np
import subprocess

from requests import get
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ws_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)
# config
from configs import update_config, config
# trainer
from engine import EpochBasedTrainer
# models
import torch
from torch.nn import functional as F
import torch.optim as optim
from models.lidarclip.model.sst import LidarEncoderSST

# dataset
from datasets.loaders import get_train_val_data_loader, get_train_dataloader, get_val_dataloader
from datasets.scan3r_lidarclip import Scan3rLidarClipDataset
from datasets.scannet_lidarclip import ScannetLidarClipDataset
# utils
from utils import common, scan3r, torch_util


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser=None):
        super().__init__(cfg, parser)
        
        # cfg
        self.cfg = cfg  
        
        # get device 
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        # get data loader
        start_time = time.time()
        self.train_dataset, self.train_loader, self.val_dataset, self.val_loader = self.getDataLoader(cfg)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        ## load large data here
        self.train_data_type = cfg.train.data_type
        self.val_scan_pcs = self.val_dataset.load_scan_pcs()
        if self.train_data_type == 'scan':
            self.train_scan_pcs = self.train_dataset.load_scan_pcs()
        elif self.train_data_type == 'depth':
            self.val_depthmap_pcs = self.val_dataset.load_depthmap_pcs() 
        
        # generate model
        self.registerModels(cfg)  
        
        # optimizer
        self.registerOptim(cfg)
        
        # loss type 
        self.loss_type = cfg.train.loss.loss_type
        self.temperature = cfg.train.loss.temperature
        
        # scheduler
        if cfg.train.optim.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, cfg.train.optim.lr_decay_steps, gamma=cfg.train.optim.lr_decay)
        elif cfg.train.optim.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.train.optim.T_max, eta_min=cfg.train.optim.lr_min,
                T_mult=cfg.train.optim.T_mult, last_epoch=-1)
        else:
            raise NotImplementedError('Scheduler {} not implemented.'.format(cfg.train.optim.scheduler))
        self.register_scheduler(scheduler)
        
        
        # log step for training
        if self.cfg.train.log_steps:
            self.log_steps = self.cfg.train.log_steps
        self.snapshot_steps = self.cfg.train.snapshot_steps
        self.logger.info('Initialisation Complete')
        
    def registerModels(self, cfg):
        # load backbone
        backbone3d_cfg_file = cfg.model.backbone3D.cfg_file
        backbone3d_pretrained_file = cfg.model.backbone3D.pretrained
        dim_3d = cfg.model.backbone3D.dim_3d
        self.backbone3d = LidarEncoderSST(backbone3d_cfg_file, dim_3d)
        self.backbone3d.load_state_dict(torch.load(backbone3d_pretrained_file), strict=False)

        # model to cuda 
        self.model = self.backbone3d
        if cfg.other.use_resume and os.path.isfile(cfg.other.resume):
            self.load_snapshot(cfg.other.resume)
        self.model.to(torch.float)
        self.model.to(self.device)
        
        self.test_params = {}
        for name, param in self.model._pooler.named_parameters():
            self.test_params[name] = param
            
        # log
        message = 'Model description:\n' + str(self.model)
        self.logger.info(message)      
        
    def after_train_step(self,epoch, iteration, data_dict, output_dict, result_dict):
        # make sure params is good 
        for name, param in self.test_params.items():
            if torch.isnan(param).any():
                self.logger.error('NaN in gradients.')
                break_point = 1

    def registerOptim(self, cfg):  
        # only optimise params that require grad
        self.params_register = list( filter(lambda p: p.requires_grad, self.model.parameters()) )
        self.params_register_ids = list(map(id, self.params_register))
        self.params = [{'params' :  self.params_register}]
        self.optimizer = optim.Adam(self.params, lr=cfg.train.optim.lr, 
                                    weight_decay=cfg.train.optim.weight_decay)

    def getDataLoader(self, cfg):
        if cfg.data.name == 'Scan3R':
            dataset = Scan3rLidarClipDataset
        elif cfg.data.name == 'Scannet':
            dataset = ScannetLidarClipDataset
        else:
            raise NotImplementedError('Dataset {} not implemented.'.format(cfg.data.name))
        # train_dataloader, val_dataloader = get_train_val_data_loader(cfg, dataset)
        train_dataset, train_dataloader = get_train_dataloader(cfg, dataset)
        val_dataset, val_dataloader = get_val_dataloader(cfg, dataset)
        
        return train_dataset, train_dataloader, val_dataset, val_dataloader

    def model_forward(self, pcs):
        embeddings_3D_pcs = self.model(pcs)
        return embeddings_3D_pcs
    
    def sim_calculation(self, embeddings_2D, embeddings_3D):
            sim_mse = -torch.norm(embeddings_2D - embeddings_3D, dim=1) # l2 sim
            embeddings_3D_norm = F.normalize(embeddings_3D, dim=1)
            embeddings_2D_norm = F.normalize(embeddings_2D, dim=1)
            sim_cos = torch.mm(embeddings_2D_norm, embeddings_3D_norm.permute(1, 0)) # cos sim
            return sim_mse, sim_cos
        
    def sim_calculation_eval(self, embeddings_2D, embeddings_3D):
        if self.loss_type == 'mse':
            sim_mse = -torch.norm(embeddings_2D - embeddings_3D, dim=1) # l2 sim
            return sim_mse, None
        else:
            embeddings_3D_norm = F.normalize(embeddings_3D, dim=1)
            embeddings_2D_norm = F.normalize(embeddings_2D, dim=1)
            sim_cos = torch.mm(embeddings_2D_norm, embeddings_3D_norm.permute(1, 0)) # cos sim
            return None, sim_cos
    
    def loss_valid(self, loss):
        if loss.isnan().all():
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        else:
            loss = loss[~loss.isnan()]
            return loss.mean()

    def train_step(self, epoch, iteration, data_dict):
        
        if data_dict is None:
            return None, None
        
        embeddings = {}
        if self.train_data_type == 'scan':
            pcs_batch = torch_util.to_cuda([self.train_scan_pcs[scan_id].contiguous() for scan_id in data_dict['scan_ids']])
        elif self.train_data_type == 'depth':
            pcs_batch = torch_util.to_cuda(data_dict['pcs_batch'])
        else:
            raise NotImplementedError('Data type {} not implemented.'.format(self.train_data_type))
        embeddings_3D,_ = self.model_forward(pcs_batch)
        # tuple to tensor 
        embeddings_2D = data_dict['img_features']
        sim_mse, sim_cos = self.sim_calculation(embeddings_2D, embeddings_3D)
        embeddings = {'embeddings_3D': embeddings_3D}
        
        e2j_matrix = data_dict['e2j_matrix']
        sim_E1i_E1j = torch.exp(torch.diag(sim_cos, 0) / self.temperature) 
        sim_E1i_E2j = torch.exp( (e2j_matrix * sim_cos) / self.temperature ).sum(dim=-1)# (N_P)
        loss_N_pair = -torch.log(sim_E1i_E1j / (sim_E1i_E1j + sim_E1i_E2j + 1e-8)+ 1e-8) 
        loss_mse = -sim_mse
        
        loss_dict = {}
        loss = loss_mse if self.loss_type == 'mse' else loss_N_pair
        loss_dict['loss'] = self.loss_valid(loss)
        # loss_dict['loss_mse'] = self.loss_valid(loss_mse)
        # loss_dict['loss_N_pair'] = self.loss_valid(loss_N_pair)
        return embeddings, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        # not grad 
        with torch.no_grad():
            embeddings = {}
            if self.train_data_type == 'scan':
                pcs_batch = torch_util.to_cuda([self.val_scan_pcs[scan_id].contiguous() for scan_id in data_dict['scan_ids']])
            elif self.train_data_type == 'depth':
                pcs_batch = torch_util.to_cuda(data_dict['pcs_batch'])
            else:
                raise NotImplementedError('Data type {} not implemented.'.format(self.train_data_type))
            embeddings_3D,_ = self.model_forward(pcs_batch)
            embeddings_2D = data_dict['img_features']
            sim_mse, sim_cos = self.sim_calculation(embeddings_2D, embeddings_3D)
            embeddings = {'embeddings_3D': embeddings_3D}
            
            e2j_matrix = data_dict['e2j_matrix']
            sim_E1i_E1j = torch.exp(torch.diag(sim_cos, 0) / self.temperature) 
            sim_E1i_E2j = torch.exp( (e2j_matrix * sim_cos) / self.temperature ).sum(dim=-1)# (N_P)
            loss_N_pair = -torch.log(sim_E1i_E1j / (sim_E1i_E1j + sim_E1i_E2j + 1e-8)+ 1e-8) 
            loss_mse = -sim_mse
            
            loss_dict = {}
            loss = loss_mse if self.loss_type == 'mse' else loss_N_pair
            loss_dict['loss'] = self.loss_valid(loss)
            loss_dict['loss_mse'] = self.loss_valid(loss_mse)
            loss_dict['loss_N_pair'] = self.loss_valid(loss_N_pair)
            
            # if epoch % self.snapshot_steps == 1:
            # retrieval_result_depth_map = self.room_retrieval_depthmap(data_dict)
            retrieval_result_scan = self.room_retrieval_scan(data_dict)
            # loss_dict.update(retrieval_result_depth_map)
            loss_dict.update(retrieval_result_scan)
            return embeddings, loss_dict
    
    def room_retrieval_depthmap(self, data_dict):
        
        # room retrieval with depth maps
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_D".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_D".format(k): 0. for k in top_k_list}
        retrieval_time_non_temporal = 0.
        retrieval_time_temporal = 0.
        
        for batch_i in range(batch_size):
            img_feature = data_dict['img_features'][batch_i].unsqueeze(0)
            cur_scan_id = data_dict['scan_ids'][batch_i]
            
            # non-temporal retrieval
            room_score_depthmap = {}
            scans_depthmaps_embeddings = {}
            ## get cur depth scan embeddings
            # curr_scan_depthmap_pcs = data_dict['curr_scan_depthmap_pcs_list'][batch_i]
            curr_scan_depthmap_pcs = torch_util.to_cuda(self.val_depthmap_pcs[cur_scan_id]) 
            depthmaps_pcs = [depthmap_pc.contiguous() for frame_idx, depthmap_pc in curr_scan_depthmap_pcs.items()]
            scans_depthmaps_embeddings[cur_scan_id], _ = self.model_forward(depthmaps_pcs)
            ## get candidate depth scan embeddings
            # candidate_depthmap_pcs = data_dict['candidate_depthmap_pcs_list'][batch_i]
            candidate_scans = data_dict['candidate_scan_ids_list'][batch_i]
            candidate_depthmap_pcs = {scan_id: torch_util.to_cuda(self.val_depthmap_pcs[scan_id]) for scan_id in candidate_scans}
            for candidate_scan_id, depthmap_pcs in candidate_depthmap_pcs.items():
                depthmaps_pcs = [depthmap_pc.contiguous() for frame_idx, depthmap_pc in depthmap_pcs.items()]
                scans_depthmaps_embeddings[candidate_scan_id], _ = self.model_forward(depthmaps_pcs)
                
            ## calculate similarity
            start_time = time.time()
            for candidate_scan_id, candidate_scan_embeddings in scans_depthmaps_embeddings.items():
                sim = self.sim_calculation(img_feature, candidate_scan_embeddings)
                room_score_depthmap[candidate_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_depthmap.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if cur_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_non_temporal["R@{}_NT_D".format(k)] += 1
            # record time
            retrieval_time_non_temporal += time.time() - start_time
                    
            # temporal retrieval
            temporal_scan_id = data_dict['temporal_scan_id_list'][batch_i]
            ## get temporal depth scan embeddings
            # temporal_scan_depthmap_pcs = data_dict['temporal_scan_depthmap_pcs_list'][batch_i]
            temporal_scan_depthmap_pcs = torch_util.to_cuda(self.val_depthmap_pcs[temporal_scan_id])
            depthmaps_pcs = [depthmap_pc.contiguous() for frame_idx, depthmap_pc in temporal_scan_depthmap_pcs.items()]
            scans_depthmaps_embeddings[temporal_scan_id],_ = self.model_forward(depthmaps_pcs)
            ## remove cur scan embeddings
            if cur_scan_id != temporal_scan_id:
                scans_depthmaps_embeddings.pop(cur_scan_id)
                room_score_depthmap.pop(cur_scan_id)
            ## calculate similarity
            start_time = time.time()
            room_score_depthmap[temporal_scan_id] = self.sim_calculation(img_feature, scans_depthmaps_embeddings[temporal_scan_id]).max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_depthmap.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if temporal_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_D".format(k)] += 1
            # record time
            retrieval_time_temporal += time.time() - start_time
            
        # average over batch
        for k in top_k_list:
            top_k_recall_temporal["R@{}_T_D".format(k)] /= 1.0*batch_size
            top_k_recall_non_temporal["R@{}_NT_D".format(k)] /= 1.0*batch_size
        retrieval_time_non_temporal = retrieval_time_non_temporal / 1.0*batch_size
        retrieval_time_temporal = retrieval_time_temporal / 1.0*batch_size
        
        result = {
            'time_T_D': retrieval_time_temporal,
            'time_NT_D': retrieval_time_non_temporal,
        }
        result.update(top_k_recall_temporal)
        result.update(top_k_recall_non_temporal)
        return result
    
    def room_retrieval_scan(self, data_dict):
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
            curr_scan_pcs = torch_util.to_cuda(self.val_scan_pcs[cur_scan_id])
            pcs = [curr_scan_pcs.contiguous()]
            scans_pcs_embeddings[cur_scan_id], _ = self.model_forward(pcs)
            ## get candidate scan embeddings
            # candidate_scans_pcs = data_dict['candidate_scan_pcs_list'][batch_i]
            candidate_scans = data_dict['candidate_scan_ids_list'][batch_i]
            candidate_scans_pcs = {scan_id: torch_util.to_cuda(self.val_scan_pcs[scan_id]) for scan_id in candidate_scans}
            for candidate_scan_id, scans_pcs in candidate_scans_pcs.items():
                pcs = [scans_pcs.contiguous()]
                scans_pcs_embeddings[candidate_scan_id], _ = self.model_forward(pcs)
            ## calculate similarity
            start_time = time.time()
            for candidate_scan_id, candidate_scan_embeddings in scans_pcs_embeddings.items():
                sim_mse, sim_cos = self.sim_calculation_eval(img_feature, candidate_scan_embeddings)
                sim = sim_mse if self.loss_type == 'mse' else sim_cos
                room_score_scans[candidate_scan_id] = sim.max().item()
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
            temporal_scan_pcs = torch_util.to_cuda(self.val_scan_pcs[temporal_scan_id])
            pcs = [temporal_scan_pcs.contiguous()]
            scans_pcs_embeddings[temporal_scan_id],_ = self.model_forward(pcs)
            ## remove cur scan embeddings
            if cur_scan_id != temporal_scan_id:
                scans_pcs_embeddings.pop(cur_scan_id)
                room_score_scans.pop(cur_scan_id)
            ## calculate similarity
            start_time = time.time()
            sim_mse, sim_cos = self.sim_calculation_eval(img_feature, scans_pcs_embeddings[temporal_scan_id])
            sim = sim_mse if self.loss_type == 'mse' else sim_cos
            room_score_scans[temporal_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if temporal_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            # record time
            retrieval_time_temporal += time.time() - start_time
            
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

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

        
def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    parser.add_argument('--log_steps', type=int, default=500, help='logging steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')

    args = parser.parse_args()
    return parser, args

def main():
    parser, args = parse_args()
    
    config_file = args.config
    cfg = update_config(config, config_file)
    
    # copy config file to out dir
    out_dir = cfg.output_dir
    command = 'cp {} {}'.format(config_file, out_dir)
    subprocess.call(command, shell=True)
    
    # train
    trainer = Trainer(cfg, parser)
    trainer.run()

if __name__ == '__main__':
    main()
