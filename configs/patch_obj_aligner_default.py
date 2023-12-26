from yacs.config import CfgNode as CN
import os.path as osp
import os

from utils import common

_C = CN()

# dataset
_C.data = CN()
_C.data.name = "Scan3R"
_C.data.root_dir = ""
_C.data.rescan = False

_C.data.img = CN()
_C.data.img.img_step = 5
_C.data.img.w = 960
_C.data.img.h = 540

_C.data.img_encoding = CN()
_C.data.img_encoding.resize_w = 1024
_C.data.img_encoding.img_rotate: True # rotate w,h for backbone GCVit
_C.data.img_encoding.patch_w: 16 # number of patchs in width
_C.data.img_encoding.patch_h: 9

_C.data.img.cross_scene = CN()
_C.data.img.cross_scene.use_cross_scene = True
_C.data.img.cross_scene.num_scenes = 10
_C.data.img.cross_scene.num_negative_samples = -1 # -1 means 

# for training
_C.train = CN()
_C.train.use_vis = False
_C.train.vis_epoch_steps = 10000

def update_config(cfg, filename, ensure_dir=True):
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(filename)
    
    # load dirs from env variables because CFG node doesn't support env variables
    Scan3R_ROOT_DIR = os.getenv('Scan3R_ROOT_DIR')
    VLSG_SPACE = os.getenv('VLSG_SPACE')
    VLSG_TRAINING_OUT_DIR = os.getenv('VLSG_TRAINING_OUT_DIR')
    RESUME_DIR = os.getenv('RESUME_DIR')
    # data root
    cfg.data.root_dir = Scan3R_ROOT_DIR
    # backbone files
    cfg.model.backbone.cfg_file = osp.join(VLSG_SPACE, cfg.model.backbone.cfg_file)
    cfg.model.backbone.pretrained = osp.join(VLSG_SPACE, cfg.model.backbone.pretrained)
    # output dir
    cfg.output_dir = VLSG_TRAINING_OUT_DIR
    # resume dir 
    cfg.other.resume = osp.join(RESUME_DIR, cfg.other.resume)
    
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