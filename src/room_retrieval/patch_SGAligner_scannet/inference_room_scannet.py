import argparse
import os 
import os.path as osp
from re import T
import time
from tracemalloc import start
import comm
from matplotlib import patches
import numpy as np 
import sys
import subprocess
import tqdm

from requests import patch
from sympy import N
from yaml import scan

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ws_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)
# utils
from utils import common
from utils import torch_util
# from utils import visualisation
# config
from configs import update_config_room_retrival, config
# tester
from engine.single_tester import SingleTester
from utils.summary_board import SummaryBoard
# models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_backbone
from mmcv import Config
# from models.GCVit.models import gc_vit
from models.patch_SGIE_aligner import PatchSGIEAligner
# dataset
from datasets.loaders import get_test_dataloader, get_val_dataloader
from datasets.scannet_objpair import ScannetPatchObjDataset

def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)

def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)

# get max, min number of candidate scans
def get_max_min_num_candidates(datadict, max = None, min = None):
    batch_size = len(datadict['assoc_data_dict'])
    for batch_i in range(batch_size):
        num_candidates = len(datadict['assoc_data_dict'][batch_i]['candata_scan_obj_idxs'])
        if max is None:
            max = num_candidates
        if min is None:
            min = num_candidates
        if num_candidates > max:
            max = num_candidates
        if num_candidates < min:
            min = num_candidates
    return max, min

# use PathObjAligner for room retrieval
class RoomRetrivalScore():
    def __init__(self, cfg):
        
        # cfg
        self.cfg = cfg 
        self.method_name = cfg.val.room_retrieval.method_name
        self.temporal = cfg.data.temporal
        
        # dataloader
        start_time = time.time()
        val_dataset, val_data_loader = get_val_dataloader(cfg, Dataset = ScannetPatchObjDataset)
        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset = ScannetPatchObjDataset)

        # register dataloader
        self.val_data_loader = val_data_loader
        self.val_dataset = val_dataset
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        
        # get device 
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        # model
        self.registerPatchObjectAlignerFromCfg(cfg)
        self.model.eval()
        self.use_tf_idf = cfg.data.cross_scene.use_tf_idf
        
        # results
        self.val_room_retrieval_summary = SummaryBoard(adaptive=True)
        self.test_room_retrieval_summary = SummaryBoard(adaptive=True)
        self.val_room_retrieval_record = {}
        self.test_room_retrieval_record = {}
        
        # files
        self.output_dir = osp.join(cfg.output_dir, self.method_name)
        common.ensure_dir(self.output_dir)

    def load_snapshot(self, snapshot, fix_prefix=True):
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        # Load model
        model_dict = state_dict['model']
        self.model.load_state_dict(model_dict, strict=False)

    def registerPatchObjectAlignerFromCfg(self, cfg):
        if cfg.data.img_encoding.use_feature:
            backbone = None
        else:
            backbone_cfg_file = cfg.model.backbone.cfg_file
            # ugly hack to load pretrained model, maybe there is a better way
            backbone_cfg = Config.fromfile(backbone_cfg_file)
            backbone_pretrained_file = cfg.model.backbone.pretrained
            backbone_cfg.model['backbone']['pretrained'] = backbone_pretrained_file
            backbone = build_backbone(backbone_cfg.model['backbone'])
        
        # get patch object aligner
        # get patch object aligner
        drop = cfg.model.other.drop
        ## 2Dbackbone
        num_reduce = cfg.model.backbone.num_reduce
        backbone_dim = cfg.model.backbone.backbone_dim
        img_rotate = cfg.data.img_encoding.img_rotate
        ## scene graph encoder
        sg_modules = cfg.sgaligner.modules
        sg_rel_dim = cfg.sgaligner.model.rel_dim
        attr_dim = cfg.sgaligner.model.attr_dim
        img_patch_feat_dim = cfg.sgaligner.model.img_patch_feat_dim
        if hasattr(cfg.sgaligner.model, 'multi_view_aggregator'):
            multi_view_aggregator = cfg.sgaligner.model.multi_view_aggregator
        else:
            multi_view_aggregator = None
            
        ## encoders
        patch_hidden_dims = cfg.model.patch.hidden_dims
        patch_encoder_dim = cfg.model.patch.encoder_dim
        patch_gcn_layers = cfg.model.patch.gcn_layers
        obj_embedding_dim = cfg.model.obj.embedding_dim
        obj_embedding_hidden_dims = cfg.model.obj.embedding_hidden_dims
        obj_encoder_dim = cfg.model.obj.encoder_dim
        img_emb_dim = cfg.sgaligner.model.img_emb_dim
        ## temporal 
        self.use_temporal = cfg.train.loss.use_temporal
        ## global descriptor
        self.use_global_descriptor = cfg.train.loss.use_global_descriptor
        self.global_descriptor_dim = cfg.model.global_descriptor_dim
        
        self.model = PatchSGIEAligner(backbone,
                                num_reduce,
                                backbone_dim,
                                img_rotate, 
                                patch_hidden_dims,
                                patch_encoder_dim,
                                patch_gcn_layers,
                                obj_embedding_dim,
                                obj_embedding_hidden_dims,
                                obj_encoder_dim,
                                sg_modules,
                                sg_rel_dim,
                                attr_dim,
                                img_patch_feat_dim,
                                drop,
                                self.use_temporal,
                                self.use_global_descriptor,
                                self.global_descriptor_dim,
                                multi_view_aggregator = multi_view_aggregator,
                                img_emb_dim = img_emb_dim,)
        
        
        # load snapshot if required
        if cfg.other.use_resume:
            assert os.path.isfile(cfg.other.resume), 'Snapshot not found.'
            self.load_snapshot(cfg.other.resume)
        # model to cuda 
        self.model.to(self.device)
        self.model.eval()

    def model_forward(self, data_dict):
        # assert self.cfg.data.img_encoding.use_feature != True, \
        #     'To measure runtime, please dont use pre-calculated features.'
        
        # image features, image by image for fair time comparison
        batch_size = data_dict['batch_size']
        forward_time = 0.
        patch_features_batch = None
        for i in range(batch_size):
            with torch.no_grad():
                start_time = time.time()
                if self.cfg.data.img_encoding.use_feature:
                    features = data_dict['patch_features'][i:i+1]
                else:
                    images = data_dict['images'] # (B, H, W, C)
                    image = images[i:i+1]
                    image = _to_channel_first(image)
                    features = self.model.backbone(image)[-1]
                    features = _to_channel_last(features)
                patch_features = self.model.reduce_layers(features)
                patch_features = self.model.patch_encoder(patch_features)
                forward_time += time.time() - start_time
                patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
            patch_features_batch = patch_features if patch_features_batch is None \
                else torch.cat([patch_features_batch, patch_features], dim=0)
        
        # object features
        obj_3D_embeddings = self.model.forward_scene_graph(data_dict)  # (O, C*)
        obj_3D_embeddings = self.model.obj_embedding_encoder(obj_3D_embeddings) # (O, C*)
        obj_3D_embeddings_norm = F.normalize(obj_3D_embeddings, dim=-1)
        return patch_features_batch, obj_3D_embeddings_norm, forward_time
    
    def room_retrieval_dict(self, data_dict, dataset, room_retrieval_record, record_retrieval = False):
        result = {}
        # room retrieval with scan point cloud
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_S".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_S".format(k): 0. for k in top_k_list}
        retrieval_time_temporal = 0.
        retrieval_time_non_temporal = 0.
        img_forward_time = 0.
        matched_obj_idxs, matched_obj_idxs_temp = None, None
        
        obj_ids = data_dict['scene_graphs']['obj_ids']
        obj_ids_cpu = torch_util.release_cuda_torch(obj_ids)
        
        # get embeddings
        patch_features_batch, obj_3D_embeddings_norm, forward_time = \
                self.model_forward(data_dict)
        for batch_i in range(batch_size):
            patch_features = patch_features_batch[batch_i]
            # non-temporal
            room_score_scans_NT = {}
            assoc_data_dict = data_dict['assoc_data_dict'][batch_i]
            candidates_obj_sg_idxs = assoc_data_dict['scans_sg_obj_idxs']
            candata_scan_obj_idxs = assoc_data_dict['candata_scan_obj_idxs']
            if self.use_tf_idf:
                reweight_matrix_scans = assoc_data_dict['reweight_matrix_scans']
                reweight_matrix_scans = torch_util.release_cuda_torch(reweight_matrix_scans)
            target_scan_id = data_dict['target_scan_ids'][batch_i]
            candidates_objs_embeds_scan = obj_3D_embeddings_norm[candidates_obj_sg_idxs]
            ## start room retrieval in cpu
            patch_features_cpu = patch_features.cpu()
            candata_scan_obj_idxs_cpu = torch_util.release_cuda_torch(candata_scan_obj_idxs)
            obj_3D_embeddings_norm_cpu_scan = torch_util.release_cuda_torch(candidates_objs_embeds_scan)
            candidates_obj_embeds = {
                candidate_scan_id: obj_3D_embeddings_norm_cpu_scan[candata_scan_obj_idxs_cpu[candidate_scan_id]] \
                for candidate_scan_id in candata_scan_obj_idxs_cpu}
            start_time = time.time()
            patch_features_cpu_norm = F.normalize(patch_features_cpu, dim=1)
            for candidate_scan_id in candidates_obj_embeds:
                candidate_obj_embeds = candidates_obj_embeds[candidate_scan_id]
                patch_obj_sim = patch_features_cpu_norm@candidate_obj_embeds.T
                if self.use_tf_idf:
                    reweight_obj_matrixs = reweight_matrix_scans[candidate_scan_id] + 0.5
                    matched_candidate_objs_idxs = patch_obj_sim.argmax(dim=1)
                    matched_sim = patch_obj_sim.gather(1, matched_candidate_objs_idxs.unsqueeze(1)).squeeze(1)
                    reweight_patch_obj_sim = matched_sim * reweight_obj_matrixs[matched_candidate_objs_idxs]
                    score = reweight_patch_obj_sim.sum().item() / reweight_obj_matrixs.sum().item()
                else:
                    matched_candidate_obj_sim = torch.max(patch_obj_sim, dim=1)[0]
                    score = matched_candidate_obj_sim.sum().item()
                room_score_scans_NT[candidate_scan_id] = score
            room_sorted_by_scores_NT =  [item[0] for item in sorted(room_score_scans_NT.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if target_scan_id in room_sorted_by_scores_NT[:k]:
                    top_k_recall_non_temporal["R@{}_NT_S".format(k)] += 1
            retrieval_time_non_temporal += time.time() - start_time
            
            matched_obj_idxs = (patch_features_cpu_norm @ candidates_obj_embeds[target_scan_id].T).argmax(dim=1)
            obj_ids_cpu_scan = obj_ids_cpu[candidates_obj_sg_idxs.cpu()][candata_scan_obj_idxs_cpu[target_scan_id]]
            matched_obj_obj_ids = obj_ids_cpu_scan[matched_obj_idxs]
            
            # retrieva_record
            scan_id = data_dict['scan_ids'][batch_i]
            if scan_id not in room_retrieval_record:
                room_retrieval_record[scan_id] = {'frames_retrieval': {}}
                room_retrieval_record[scan_id]['candidates_scan_ids'] = dataset.candidate_scans[scan_id]
                room_retrieval_record[scan_id]['obj_ids'] = dataset.scene_graphs[scan_id]['obj_ids']
            frame_idx = data_dict['frame_idxs'][batch_i]
            frame_retrieval = {
                'frame_idx': frame_idx,
                'matched_obj_obj_ids': matched_obj_obj_ids,
                'room_score_scans_NT': room_score_scans_NT,
            }
            room_retrieval_record[scan_id]['frames_retrieval'][frame_idx] = frame_retrieval

        # average over batch
        for k in top_k_list:
            top_k_recall_non_temporal["R@{}_NT_S".format(k)] /= 1.0*batch_size
        retrieval_time_temporal = retrieval_time_temporal / (1.0*batch_size)
        retrieval_time_non_temporal = retrieval_time_non_temporal / (1.0*batch_size)
        img_forward_time = forward_time / (1.0*batch_size)
        
        result = {
            'img_forward_time': img_forward_time,
            'time_NT_S': retrieval_time_non_temporal,
        }
        result.update(top_k_recall_non_temporal)
        return result

    def room_retrieval_val(self):
        # val 
        # record number of candidates
        max_num_candidates, min_num_candidates = None, None
        with torch.no_grad():
            data_dicts = tqdm.tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader))
            for iteration, data_dict in data_dicts:
                data_dict = torch_util.to_cuda(data_dict)
                result = self.room_retrieval_dict(data_dict, self.val_dataset, self.val_room_retrieval_record, True)
                self.val_room_retrieval_summary.update_from_result_dict(result)
                torch.cuda.empty_cache()
                
                max_num_candidates, min_num_candidates = get_max_min_num_candidates(
                    data_dict, max_num_candidates, min_num_candidates)
            
        val_items = self.val_room_retrieval_summary.tostringlist()
        num_candidates_string = 'Val: Max num candidates: {}, Min num candidates: {}'.format(
            max_num_candidates, min_num_candidates)
        val_items.append(num_candidates_string)
        
        # write metric to file
        val_file = osp.join(self.output_dir, 'val_result.txt')
        common.write_to_txt(val_file, val_items)
        # write retrieval record to file
        retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_val.pkl')
        common.write_pkl_data(self.val_room_retrieval_record, retrieval_record_file)
        
        # test 
        # record number of candidates
        max_num_candidates, min_num_candidates = None, None
        with torch.no_grad():
            data_dicts = tqdm.tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader))
            for iteration, data_dict in data_dicts:
                data_dict = torch_util.to_cuda(data_dict)
                result = self.room_retrieval_dict(data_dict, self.test_dataset,self.test_room_retrieval_record, True)
                self.test_room_retrieval_summary.update_from_result_dict(result)
                torch.cuda.empty_cache()
                
                max_num_candidates, min_num_candidates = get_max_min_num_candidates(
                    data_dict, max_num_candidates, min_num_candidates)
                
        test_items = self.test_room_retrieval_summary.tostringlist()
        num_candidates_string = 'Test: Max num candidates: {}, Min num candidates: {}'.format(
            max_num_candidates, min_num_candidates)
        test_items.append(num_candidates_string)
        
        # write metric to file
        test_file = osp.join(self.output_dir, 'test_result.txt')
        common.write_to_txt(test_file, test_items)
        # write retrieval record to file
        retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_test.pkl')
        common.write_pkl_data(self.test_room_retrieval_record, retrieval_record_file)
            

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    
    cfg = update_config_room_retrival(config, args.config, ensure_dir=True)
    
    # copy config file to out dir
    out_dir = osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name)
    common.ensure_dir(out_dir)
    command = 'cp {} {}'.format(args.config, out_dir)
    subprocess.call(command, shell=True)

    tester = RoomRetrivalScore(cfg)
    tester.room_retrieval_val()
    breakpoint = 0

if __name__ == '__main__':
    main()