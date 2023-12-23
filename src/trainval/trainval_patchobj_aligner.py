import argparse
import time
import os
from collections import OrderedDict
import sys
import subprocess
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ws_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(ws_dir)
# config
from configs import update_config, config
# trainer
from engine import EpochBasedTrainer
from datasets.loaders import get_train_val_data_loader
# models
import torch
import torch.optim as optim
from mmdet.models import build_backbone
from mmcv import Config
# from models.GCVit.models import gc_vit
from models.patch_obj_aligner import PatchObjectAligner
from models.loss import ICLLoss
from models.path_obj_pair_visualizer import PatchObjectPairVisualizer



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
        self.train_loader, self.val_loader = self.getDataLoader(cfg)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        
        # generate model
        self.registerPatchObjectAlignerFromCfg(cfg)  
        
        # optimizer
        self.registerOptim(cfg)
        
        # generate loss
        self.loss = ICLLoss()
        
        # log step for training
        if self.cfg.train.log_steps:
            self.log_steps = self.cfg.train.log_steps
        self.snapshot_steps = self.cfg.train.snapshot_steps
        self.logger.info('Initialisation Complete')
        
        # register visualiser
        self.registerVisuliser(cfg)
        
    def registerPatchObjectAlignerFromCfg(self, cfg):
        # load backbone
        backbone_cfg_file = cfg.model.backbone.cfg_file
        # ugly hack to load pretrained model, maybe there is a better way
        backbone_cfg = Config.fromfile(backbone_cfg_file)
        backbone_pretrained_file = cfg.model.backbone.pretrained
        backbone_cfg.model['backbone']['pretrained'] = backbone_pretrained_file
        backbone = build_backbone(backbone_cfg.model['backbone'])
        
        # get patch object aligner
        num_reduce = cfg.model.backbone.num_reduce
        backbone_dim = cfg.model.backbone.backbone_dim
        img_rotate = cfg.data.img_encoding.img_rotate
        
        patch_hidden_dims = cfg.model.patch.hidden_dims
        patch_encoder_dim = cfg.model.patch.encoder_dim
        
        obj_embedding_dim = cfg.model.obj.embedding_dim
        obj_embedding_hidden_dims = cfg.model.obj.embedding_hidden_dims
        obj_encoder_dim = cfg.model.obj.encoder_dim
        
        drop = cfg.model.other.drop
        
        self.model = PatchObjectAligner(backbone,
                                num_reduce,
                                backbone_dim,
                                img_rotate, 
                                patch_hidden_dims,
                                patch_encoder_dim,
                                obj_embedding_dim,
                                obj_embedding_hidden_dims,
                                obj_encoder_dim,
                                drop)
        # load snapshot if required
        if cfg.other.use_resume and os.path.isfile(cfg.other.resume):
            self.load_snapshot(cfg.other.resume)
        # model to cuda 
        self.model.to(self.device)
        
        # freeze backbone params if required
        self.free_backbone_epoch = cfg.train.optim.free_backbone_epoch
        if (self.free_backbone_epoch > 0):
            self.freezeBackboneParams()
        
        # log
        message = 'Model description:\n' + str(self.model)
        self.logger.info(message)      

    def registerOptim(self, cfg):  
        # only optimise params that require grad
        self.params = [{'params' : list( filter(lambda p: p.requires_grad, self.model.parameters()) ) }]
        self.optimizer = optim.Adam(self.params, lr=cfg.train.optim.lr, 
                                    weight_decay=cfg.train.optim.weight_decay)

    def registerVisuliser(self, cfg):
        if cfg.train.use_vis:
            self.result_visualizer = PatchObjectPairVisualizer(cfg)
        else:
            self.result_visualizer = None

    def freezeBackboneParams(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
    def defreezeBackboneParams(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        # update optimiser
        bacbone_params = [{'params' : list( filter(lambda p: p.requires_grad, self.model.backbone.parameters()) ) }]
        self.optimizer.add_param_group(bacbone_params)

    def getDataLoader(self, cfg):
        train_dataloader, val_dataloader = get_train_val_data_loader(cfg)
        return train_dataloader, val_dataloader

    def train_step(self, epoch, iteration, data_dict):
        embeddings = self.model(data_dict)
        loss_dict = self.loss(embeddings, data_dict)
        return embeddings, loss_dict
    
    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        # visualize result and save
        if self.cfg.train.use_vis and epoch % self.cfg.train.vis_epoch_steps == 0:
            self.result_visualizer.visualize(data_dict, output_dict, epoch)

    def val_step(self, epoch, iteration, data_dict):
        embeddings = self.model(data_dict)
        loss_dict = self.loss(embeddings, data_dict)
        return embeddings, loss_dict

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        self.loss.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        self.loss.train()
        torch.set_grad_enabled(True)
        
    def after_train_epoch(self, epoch):
        if epoch > self.cfg.train.optim.free_backbone_epoch:
            self.defreezeBackboneParams()
        
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
