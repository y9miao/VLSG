import argparse
from math import isnan
import time
import os
from collections import OrderedDict
import sys
import numpy as np
import subprocess
from sklearn import metrics
from torch import nn
from tqdm import tqdm
import itertools

from requests import get
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ws_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)
# config
from configs import update_config, config
# trainer
from engine import EpochBasedTrainer
from utils.timer import Timer
from utils.common import get_log_string
from utils.summary_board import SummaryBoard
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
from datasets.loaders import get_train_val_data_loader, get_train_dataloader, get_val_dataloader
from datasets.scan3r_liploc import Scan3rLipLocDataset
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
        
        # generate model
        self.registerModels(cfg)  
        
        # optimizer
        self.registerOptim(cfg)
        
        # loss type 
        self.loss_type = cfg.train.loss.loss_type
        self.temperature = cfg.train.loss.temperature        
        
        # log step for training
        if self.cfg.train.log_steps:
            self.log_steps = self.cfg.train.log_steps
        self.snapshot_steps = self.cfg.train.snapshot_steps
        self.logger.info('Initialisation Complete')
        
    def registerModels(self, cfg):
        # load backbone
        liploc_pretrained_file = cfg.model.backbone3D.pretrained
        self.liploc_model = LipLocModel.Model(LipLoc_CFG)
        self.liploc_model.load_state_dict(torch.load(liploc_pretrained_file), strict=True)

        # model to cuda 
        self.model = self.liploc_model
        if cfg.other.use_resume and os.path.isfile(cfg.other.resume):
            self.load_snapshot(cfg.other.resume)
        self.model.to(torch.float)
        self.model.to(self.device)
        
        self.test_params = {}
        for name, param in self.model.named_parameters():
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
        params = [
            {
                "params": self.model.encoder_camera.parameters(), 
                "lr": LipLoc_CFG.text_encoder_lr
            },
            {
                "params": self.model.encoder_lidar.parameters(),
                "lr": LipLoc_CFG.image_encoder_lr
            },
            {
                "params": itertools.chain(self.model.projection_lidar.parameters(), 
                                          self.model.projection_camera.parameters()), 
                "lr": LipLoc_CFG.head_lr, 
                "weight_decay": LipLoc_CFG.weight_decay
            }
        ]
        self.optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=LipLoc_CFG.patience, factor=LipLoc_CFG.factor
        )
        self.register_scheduler(lr_scheduler)

    def getDataLoader(self, cfg):
        dataset = Scan3rLipLocDataset
        # train_dataloader, val_dataloader = get_train_val_data_loader(cfg, dataset)
        # train_dataloader, val_dataloader = get_train_val_data_loader(cfg, dataset)
        train_dataset, train_dataloader = get_train_dataloader(cfg, dataset)
        val_dataset, val_dataloader = get_val_dataloader(cfg, dataset)
        return train_dataset, train_dataloader, val_dataset, val_dataloader

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
        
    def loss_calculation(self, embeddings, data_dict):
        def cross_entropy(preds, targets, reduction='none'):
            log_softmax = nn.LogSoftmax(dim=-1)
            loss = (-targets * log_softmax(preds)).sum(1)
            if reduction == "none":
                return loss
            elif reduction == "mean":
                return loss.mean()
        embeddings_3D = embeddings['lidar_features']
        embeddings_2D = embeddings['camera_features']
        # calculate similarity
        logits = (embeddings_3D @ embeddings_2D.T) / self.temperature
        camera_similarity = embeddings_2D @ embeddings_2D.T
        lidar_similarity = embeddings_3D @ embeddings_3D.T
        # calculate loss
        targets = F.softmax(
            (camera_similarity + lidar_similarity) / 2 * self.temperature, dim=-1
        )
        lidar_loss = cross_entropy(logits, targets, reduction='none')
        camera_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (camera_loss + lidar_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
    def loss_valid(self, loss):
        if loss.isnan().all():
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        else:
            loss = loss[~loss.isnan()]
            return loss.mean()

    def train_step(self, epoch, iteration, data_dict):
        if data_dict is None:
            return None, None
        
        embeddings = self.model_forward(data_dict)
        loss = self.loss_calculation(embeddings, data_dict)
        loss_dict = {'loss': loss}
        return embeddings, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        # not grad 
        with torch.no_grad():
            embeddings = self.model_forward(data_dict)
            loss = self.loss_calculation(embeddings, data_dict)
            retrieval_result_scan = self.room_retrieval_scan(data_dict)
            loss_dict = {'loss':loss}
            loss_dict.update(retrieval_result_scan)
            return embeddings, loss_dict
    
    def room_retrieval_scan(self, data_dict):
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
            cur_range_img =  self.val_dataset.getRangeImagesTensor([cur_scan_id])
            cur_range_img = torch_util.to_cuda(cur_range_img)
            scans_embeddings[cur_scan_id] = self.forward_lidar(cur_range_img).squeeze(0)
            ## get candidate scan embeddings
            candidate_scans = data_dict['candidate_scan_ids_list'][batch_i]

            for candidate_scan_id in candidate_scans:
                range_img = self.val_dataset.getRangeImagesTensor([candidate_scan_id])
                range_img = torch_util.to_cuda(range_img)
                scans_embeddings[candidate_scan_id] = self.forward_lidar(range_img).squeeze(0)
            ## calculate similarity
            cur_img_embeddings = self.forward_camera(cur_img).squeeze(0)
            for candidate_scan_id, candidate_scan_embeddings in scans_embeddings.items():
                sim = candidate_scan_embeddings@cur_img_embeddings.T
                room_score_scans[candidate_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if cur_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_non_temporal["R@{}_NT_S".format(k)] += 1
                    
            # temporal retrieval
            temporal_scan_id = data_dict['temporal_scan_id_list'][batch_i]
            ## get temporal depth scan embeddings
            temporal_range_img = self.val_dataset.getRangeImagesTensor([temporal_scan_id])
            temporal_range_img = torch_util.to_cuda(temporal_range_img)
            scans_embeddings[temporal_scan_id] = self.forward_lidar(temporal_range_img).squeeze(0)
            ## remove cur scan embeddings
            scans_embeddings.pop(cur_scan_id)
            room_score_scans.pop(cur_scan_id)
            ## calculate similarity
            sim = scans_embeddings[temporal_scan_id]@cur_img_embeddings.T
            room_score_scans[temporal_scan_id] = sim.max().item()
            ## select top k similar scans
            room_sorted_by_scores =  [item[0] for item in sorted(room_score_scans.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if temporal_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            
        # average over batch
        for k in top_k_list:
            top_k_recall_temporal["R@{}_T_S".format(k)] /= 1.0*batch_size
            top_k_recall_non_temporal["R@{}_NT_S".format(k)] /= 1.0*batch_size
        retrieval_time_temporal = retrieval_time_temporal / 1.0*batch_size
        retrieval_time_non_temporal = retrieval_time_non_temporal / 1.0*batch_size
        
        result = {}
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

    def train_epoch(self):
        if self.distributed: 
            self.train_loader.sampler.set_epoch(self.epoch)
        
        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)

        for iteration, data_dict in enumerate(self.train_loader):
            self.inner_iteration = iteration + 1
            self.iteration += 1
            data_dict = torch_util.to_cuda(data_dict)
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            # forward
            output_dict, result_dict = self.train_step(self.epoch, self.inner_iteration, data_dict)
            # backward & optimization
            result_dict['loss'].backward(retain_graph=True)
            self.after_backward(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.check_gradients(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            self.optimizer_step(self.epoch)

            # after training
            self.timer.add_process_time()
            self.after_train_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            lr_dict = {'lr': self.get_lr()}
            self.summary_board.update_from_result_dict(lr_dict, 1)

            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event('train', summary_dict, self.iteration)
            
            torch.cuda.empty_cache()
            
        
        
        self.after_train_epoch(self.epoch)
        message = get_log_string(self.summary_board.summary(), epoch=self.epoch, timer=self.timer)
        self.logger.critical(message)
        # snapshot
        if self.epoch % self.snapshot_steps == 0:
            self.save_snapshot(f'epoch-{self.epoch}.pth.tar')

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm(enumerate(self.val_loader), total=total_iterations)

        for iteration, data_dict in pbar:
            self.inner_iteration = iteration + 1
            data_dict = torch_util.to_cuda(data_dict)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            output_dict, result_dict = self.val_step(self.epoch, self.inner_iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            self.after_val_step(self.epoch, self.inner_iteration, data_dict, output_dict, result_dict)
            result_dict = self.release_tensors(result_dict)
            
            summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=summary_board.summary(),
                epoch=self.epoch,
                iteration=self.inner_iteration,
                max_iteration=total_iterations,
                timer=timer,
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
            
        # scheduler
        if self.scheduler is not None:
            self.scheduler.step(metrics = summary_board.mean('loss'))
        
        summary_dict = summary_board.summary()
        message = '[Val] ' + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        val_loss = result_dict['loss']
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_snapshot(f'best_snapshot.pth.tar')

        self.logger.critical(message)
        self.write_event('val', summary_dict, self.epoch)
        self.set_train_mode()
        
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
