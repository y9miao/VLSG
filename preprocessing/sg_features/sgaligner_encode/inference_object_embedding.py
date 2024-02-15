import argparse
import os 
import os.path as osp
import time
import math
import numpy as np 
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import sys
exe_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(exe_dir)
sys.path.append(exe_dir)
sys.path.append(src_dir)
sys.path.append('.')

from engine.single_tester import SingleTester
# from engine.registration_evaluator import RegistrationEvaluator
from utils import torch_util
from aligner.sg_aligner import *
from datasets.loaders import get_val_dataloader
from datasets.scan3r_entire_scan import Scan3REntireScanDataset
from configs import config, update_config
from utils import common, scan3r, alignment

class ObjectEmbeddingGenerator:
    def __init__(self, cfg, args):
        
        # args
        self.args = args

        # Model Specific params
        self.cfg = cfg
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim

        # Metrics params
        self.alignment_thresh = cfg.model.alignment_thresh
        self.corr_score_thresh = cfg.reg_model.corr_score_thresh

        # use dataset directly
        self.dataset = Scan3REntireScanDataset(cfg, self.args.split, self.args.rescan)

        # cuda and distributed
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")

        # model 
        self.model = self.create_model()
        # self.register_model(model)
        self.model.eval()
        
    def load_snapshot(self, snapshot):
        # self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        assert 'model' in state_dict, 'No model can be loaded.'
        self.model.load_state_dict(state_dict['model'], strict=True)
        # self.logger.info('Model has been loaded.')

    def create_model(self):
        model = MultiModalSingleScanEncoder(modules = self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim).to(self.device)
        message = 'Model created'
        return model
    
    def inference_step(self, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def generate_save_embeddings(self):
        # save dir
        use_predicted = self.cfg.use_predicted
        scan_type = self.cfg.scan_type
        out_dirname = '' if scan_type == 'scan' else 'out'
        out_dirname = osp.join(out_dirname, 'predicted') if use_predicted else out_dirname
        data_dir = osp.join(self.cfg.data.root_dir, out_dirname)
        data_write_dir = osp.join(data_dir, 'files', "orig")
        embeddings_dir = osp.join(data_write_dir, 'embeddings')
        common.ensure_dir(data_write_dir)
        common.ensure_dir(embeddings_dir)
        
        # load model
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        
        # iterate scans
        num_scans = len(self.dataset)
        val_batch_size = self.cfg.val.batch_size
        num_step = int(math.ceil(num_scans/val_batch_size))
        
        for step_idx in tqdm(range(num_step)):
            # aggregate batch data
            start = step_idx*val_batch_size
            end = min( (step_idx+1)*val_batch_size, num_scans)
            data_dict = []
            for scan_i in range(start, end):
                data_dict.append(self.dataset[scan_i])
            data_dict = self.dataset.collate_fn(data_dict)
                
            # inference
            torch.cuda.synchronize()
            data_dict_cuda = torch_util.to_cuda(data_dict)
            data_inference_cuda = self.inference_step(data_dict_cuda)
            torch.cuda.synchronize()
            data_inference = torch_util.release_cuda(data_inference_cuda)
            torch.cuda.empty_cache()
            
            # save data
            objs_idx = 0
            for scan_idx, scan_id in enumerate(data_dict['scene_ids']):
                scan_embeddings = {}
                scan_id = scan_id[0]
                scan_embeddings['scan_id'] = scan_id
                # get embedding of each 3D object
                object_embeddings = {}
                obj_count = data_dict['tot_obj_count'][scan_idx]
                for obj_idx in range(objs_idx, objs_idx+obj_count):
                    obd_id = data_dict['obj_ids'][obj_idx]
                    object_embeddings[obd_id] = data_inference['joint'][obj_idx]
                scan_embeddings['obj_embeddings'] = object_embeddings
                # save
                obj_embeddings_file = osp.join(embeddings_dir, scan_id + '.pkl')
                common.write_pkl_data(scan_embeddings, obj_embeddings_file)
                objs_idx += obj_count    
        
def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--rescan', dest='rescan', default=False, action='store_true', help='get ref scan or rescan')
    parser.add_argument('--split', type=str, default="train", help='which split to generate object embeddings')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)
    embedding_generator = ObjectEmbeddingGenerator(cfg, args)
    embedding_generator.generate_save_embeddings()

if __name__ == '__main__':
    main()