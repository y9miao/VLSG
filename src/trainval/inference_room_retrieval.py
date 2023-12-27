import argparse
import os 
import os.path as osp
import time
import numpy as np 
import sys
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ws_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(ws_dir)

# config
from configs import update_config, config
# tester
from engine.single_tester import SingleTester
from utils import torch_util
from datasets.loaders import get_train_val_data_loader, get_val_dataloader
from datasets.scan3r_obj_pair_cross_scenes import PatchObjectPairCrossScenesDataSet
# models
import torch
import torch.optim as optim
from mmdet.models import build_backbone
from mmcv import Config
# from models.GCVit.models import gc_vit
from models.patch_obj_aligner import PatchObjectAligner
from models.loss import ICLLoss
from models.path_obj_pair_visualizer import PatchObjectPairVisualizer

# use PathObjAligner for room retrieval
class RoomRetrivalPatchObjAligner(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=None)
        # cfg
        self.cfg = cfg  
        # dataloader
        start_time = time.time()
        dataset, data_loader = get_val_dataloader(cfg, Dataset = PatchObjectPairCrossScenesDataSet)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        self.register_loader(data_loader)
        self.register_dataset(dataset)
        
        # model
        self.registerPatchObjectAlignerFromCfg(cfg)
        self.model.eval()
        
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
        
        # load snapshot
        assert (cfg.other.use_resume and os.path.isfile(cfg.other.resume))
        self.snap_shot = cfg.other.resume
        self.load_snapshot(cfg.other.resume)
        
        # model to cuda 
        self.model.to(self.device)
        
        # log
        message = 'Model description:\n' + str(self.model)
        self.logger.info(message) 

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict
    
    def eval_step(self, iteration, data_dict, output_dict):
        # visualise
        if self.cfg.test.vis:
            self.visualiser.visualise(data_dict, output_dict)
        return output_dict

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')

    args = parser.parse_args()
    return parser, args
    
def main():

    cfg = update_config(config, args.config, ensure_dir=True)

    tester = EVATester(cfg, parser)
    tester.run()

if __name__ == '__main__':
    main()