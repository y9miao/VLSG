from yacs.config import CfgNode as CN
import os.path as osp

from utils import common

_C = CN()

def update_config(cfg, filename, ensure_dir=True):
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(filename)
    
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