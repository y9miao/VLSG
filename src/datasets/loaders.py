from .scan3r_objpair_XTAE_SGI import PatchObjectPairXTAESGIDataSet
from utils import torch_util
import torch 
import numpy
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_val_data_loader(cfg, Dataset = PatchObjectPairXTAESGIDataSet):
    train_dataset = Dataset(cfg, split='train')
    train_dataloader = torch_util.build_dataloader(train_dataset, 
                                                   batch_size=cfg.train.batch_size, 
                                                   num_workers=cfg.train.num_workers, 
                                                   shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, 
                                                   pin_memory=True, 
                                                   drop_last=True)
    val_dataset = Dataset(cfg, split='val')
    val_dataloader = torch_util.build_dataloader(val_dataset, 
                                                 batch_size=cfg.val.batch_size, 
                                                 num_workers=cfg.val.num_workers, 
                                                 shuffle=False,
                                                 collate_fn=val_dataset.collate_fn, 
                                                 pin_memory=True, 
                                                 drop_last=True)

    return train_dataloader, val_dataloader

def get_train_dataloader(cfg, Dataset = PatchObjectPairXTAESGIDataSet):
    train_dataset = Dataset(cfg, split='train')
    train_dataloader = torch_util.build_dataloader(train_dataset, 
                                                   batch_size=cfg.train.batch_size, 
                                                   num_workers=cfg.train.num_workers, 
                                                   shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, 
                                                   pin_memory=True, 
                                                   drop_last=True)
    return train_dataset, train_dataloader

def get_val_dataloader(cfg, Dataset = PatchObjectPairXTAESGIDataSet):
    val_dataset = Dataset(cfg, split='val')
    val_dataloader = torch_util.build_dataloader(val_dataset, 
                                                 batch_size=cfg.val.batch_size, 
                                                 num_workers=cfg.val.num_workers, 
                                                 shuffle=False,
                                                collate_fn=val_dataset.collate_fn, 
                                                pin_memory=True, 
                                                drop_last=True)
    return val_dataset, val_dataloader

def get_test_dataloader(cfg, Dataset = PatchObjectPairXTAESGIDataSet):
    test_dataset = Dataset(cfg, split='test')
    test_dataloader = torch_util.build_dataloader(test_dataset, 
                                                  batch_size=cfg.val.batch_size, 
                                                  num_workers=cfg.val.num_workers, 
                                                  shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, 
                                                  pin_memory=True, 
                                                  drop_last=True)
    return test_dataset, test_dataloader

    
