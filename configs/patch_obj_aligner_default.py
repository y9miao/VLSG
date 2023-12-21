from yacs.config import CfgNode as CN
import os.path as osp
import os

from utils import common

_C = CN()

def update_config(cfg, filename, ensure_dir=True):
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(filename)
    
    # load dirs from env variables because CFG node doesn't support env variables
    Scan3R_ROOT_DIR = os.getenv('Scan3R_ROOT_DIR')
    VLSG_SPACE = os.getenv('VLSG_SPACE')
    VLSG_TRAINING_OUT_DIR = os.getenv('VLSG_TRAINING_OUT_DIR')
    # data root
    cfg.data.root_dir = Scan3R_ROOT_DIR
    # backbone files
    cfg.model.backbone.cfg_file = osp.join(VLSG_SPACE, cfg.model.backbone.cfg_file)
    cfg.model.backbone.pretrained = osp.join(VLSG_SPACE, cfg.model.backbone.pretrained)
    # output dir
    cfg.output_dir = VLSG_TRAINING_OUT_DIR
    cfg.other.resume = osp.join(VLSG_TRAINING_OUT_DIR, cfg.other.resume)
    
    if ensure_dir:
        cfg.snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
        cfg.log_dir = osp.join(cfg.output_dir, 'logs')
        cfg.event_dir = osp.join(cfg.output_dir, 'events')
        common.ensure_dir(cfg.output_dir)
        common.ensure_dir(cfg.snapshot_dir)
        common.ensure_dir(cfg.log_dir)
        common.ensure_dir(cfg.event_dir)
    cfg.freeze()
    
    return cfg