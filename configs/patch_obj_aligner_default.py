
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
_C.data.temporal = False
_C.data.resplit = False

_C.data.img = CN()
_C.data.img.img_step = 5
_C.data.img.w = 960
_C.data.img.h = 540

_C.data.img_encoding = CN()
_C.data.img_encoding.resize_w = 1024
_C.data.img_encoding.img_rotate: True # rotate w,h for backbone GCVit
_C.data.img_encoding.patch_w: 16 # number of patchs in width
_C.data.img_encoding.patch_h: 9
_C.data.img_encoding.record_feature = False
_C.data.img_encoding.use_feature = False
_C.data.img_encoding.preload_feature = False
_C.data.img_encoding.feature_dir = ''

_C.data.cross_scene = CN()
_C.data.cross_scene.use_cross_scene = False
_C.data.cross_scene.num_scenes = 0
_C.data.cross_scene.num_negative_samples = 0 # -1 means 
_C.data.cross_scene.use_tf_idf = False

_C.data.scene_graph = CN()
_C.data.scene_graph.obj_img_patch = ""
_C.data.scene_graph.obj_patch_num = 1
_C.data.scene_graph.obj_topk = 1

_C.data.auxiliary = CN()
_C.data.auxiliary.use_patch_depth = False
_C.data.auxiliary.depth_dir = ''

# model
_C.model = CN()
_C.model.backbone = CN()
_C.model.backbone.name = "GCViT"
_C.model.backbone.cfg_file = ""
_C.model.backbone.pretrained = ""
_C.model.backbone.use_pretrained = True
_C.model.backbone.num_reduce = 1
_C.model.backbone.backbone_dim = 512
_C.model.patch = CN()
_C.model.patch.hidden_dims = []
_C.model.patch.encoder_dim = 256
_C.model.obj = CN()
_C.model.obj.embedding_dim = 256
_C.model.obj.embedding_hidden_dims = []
_C.model.obj.encoder_dim = 256
_C.model.other = CN()
_C.model.other.drop = 0.0

# for training
_C.train = CN()
_C.train.gpus = 1
_C.train.precision = 16
_C.train.batch_size = 1
_C.train.num_workers = 1
_C.train.freeze_backbone = False
_C.train.use_pretrained = False
_C.train.log_steps = 1
_C.train.snapshot_steps = 100
_C.train.optim = CN()
_C.train.optim.lr = 0.001
_C.train.optim.scheduler = 'step'
## for stepLr
_C.train.optim.lr_decay = 0.95
_C.train.optim.lr_decay_steps = 10
## for CosineAnnealingLR
_C.train.optim.lr_min = 0.001
_C.train.optim.T_max = 1000
_C.train.optim.T_mult = 1
## 
_C.train.optim.weight_decay = 0.0001
_C.train.optim.max_epoch = 10000
_C.train.optim.free_backbone_epoch = 10000
_C.train.optim.grad_acc_steps = 1
## loss
_C.train.loss = CN()
_C.train.loss.use_temporal = False
_C.train.loss.loss_type = 'ICLLoss'
_C.train.loss.alpha = 0.5 # for contrastive loss
_C.train.loss.temperature = 0.1
_C.train.loss.margin = 0.1 # for triplet loss
_C.train.loss.epsilon = 1e-8 
## vis
_C.train.use_vis = False
_C.train.vis_epoch_steps = 10000
## data augmentation
_C.train.data_aug = CN()
_C.train.data_aug.use_aug = False
_C.train.data_aug.img = CN()
_C.train.data_aug.img.rotation = 0.
_C.train.data_aug.img.horizontal_flip = 0.
_C.train.data_aug.img.vertical_flip = 0.
_C.train.data_aug.img.color = 0.
_C.train.data_aug.use_aug_3D = False
_C.train.data_aug.pcs = CN()
_C.train.data_aug.pcs.granularity = [0.05]
_C.train.data_aug.pcs.magnitude = [0.]
# for validation
_C.val = CN()
_C.val.batch_size = 1
_C.val.num_workers = 1
_C.val.pretrained = ''
_C.val.room_retrieval = CN()
_C.val.room_retrieval.epsilon_th = 0.8
_C.val.room_retrieval.method_name = ''

# others
_C.other = CN()
_C.other.use_resume = False
_C.other.resume = ''
_C.other.resume_folder = ""

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
    
    if ensure_dir:
        # backbone files
        cfg.model.backbone.cfg_file = osp.join(VLSG_SPACE, cfg.model.backbone.cfg_file)
        cfg.model.backbone.pretrained = osp.join(VLSG_SPACE, cfg.model.backbone.pretrained)
        # output dir
        cfg.output_dir = VLSG_TRAINING_OUT_DIR
        # resume dir 
        cfg.other.resume = osp.join(RESUME_DIR, cfg.other.resume)
        cfg.snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
        cfg.log_dir = osp.join(cfg.output_dir, 'logs')
        cfg.event_dir = osp.join(cfg.output_dir, 'events')
        common.ensure_dir(cfg.output_dir)
        common.ensure_dir(cfg.snapshot_dir)
        common.ensure_dir(cfg.log_dir)
        common.ensure_dir(cfg.event_dir)
    cfg.freeze()
    
    return cfg

def update_config_room_retrival(cfg, filename, ensure_dir=True):
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(filename)
    
    # load dirs from env variables because CFG node doesn't support env variables
    Scan3R_ROOT_DIR = os.getenv('Scan3R_ROOT_DIR')
    VLSG_SPACE = os.getenv('VLSG_SPACE')
    ROOM_RETRIEVAL_OUT_DIR = os.getenv('ROOM_RETRIEVAL_OUT_DIR')
    RESUME_DIR = os.getenv('RESUME_DIR')
    # data root
    cfg.data.root_dir = Scan3R_ROOT_DIR
    
    if ensure_dir:
        # backbone files
        cfg.model.backbone.cfg_file = osp.join(VLSG_SPACE, cfg.model.backbone.cfg_file)
        cfg.model.backbone.pretrained = osp.join(VLSG_SPACE, cfg.model.backbone.pretrained)
        # output dir
        cfg.output_dir = ROOM_RETRIEVAL_OUT_DIR
        # resume dir 
        cfg.other.resume = osp.join(RESUME_DIR, cfg.other.resume)
        common.ensure_dir(osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name))
        cfg.log_dir = osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name, 'logs')
        common.ensure_dir(cfg.log_dir)

    cfg.freeze()
    
    return cfg