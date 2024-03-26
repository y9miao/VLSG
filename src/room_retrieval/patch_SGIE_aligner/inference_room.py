import argparse
from enum import unique
from math import e
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
from datasets.scan3r_objpair_XTAE_SGI import PatchObjectPairXTAESGIDataSet
# statistics
from utils.visualisation import RetrievalStatistics
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

# use PathObjAligner for room retrieval
class RoomRetrivalScore():
    def __init__(self, cfg):
        
        # cfg
        self.cfg = cfg 
        self.method_name = cfg.val.room_retrieval.method_name
        
        # dataloader
        start_time = time.time()
        val_dataset, val_data_loader = get_val_dataloader(cfg, Dataset = PatchObjectPairXTAESGIDataSet)
        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset = PatchObjectPairXTAESGIDataSet)
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
        self.loss_type = cfg.train.loss.loss_type
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
        if hasattr(cfg.sgaligner, 'use_pos_enc'):
            use_pos_enc = cfg.sgaligner.use_pos_enc
        else:
            use_pos_enc = False
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
                                img_emb_dim = img_emb_dim,
                                obj_img_pos_enc=use_pos_enc)
        
        # load pretrained sgaligner if required
        if cfg.sgaligner.use_pretrained:
            assert os.path.isfile(cfg.sgaligner.pretrained), 'Pretrained sgaligner not found.'
            sgaligner_dict = torch.load(cfg.sgaligner.pretrained, map_location=torch.device('cpu'))
            sgaligner_model = sgaligner_dict['model']
            # remove weights of the last layer
            sgaligner_model.pop('fusion.weight')
            self.model.sg_encoder.load_state_dict(sgaligner_dict['model'], strict=False)
        
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
                # to channel first
                patch_features = _to_channel_first(patch_features)
                patch_features = self.model.patch_gcn(patch_features)
                # to channel last
                patch_features = _to_channel_last(patch_features)
                forward_time += time.time() - start_time
                patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
            patch_features_batch = patch_features if patch_features_batch is None \
                else torch.cat([patch_features_batch, patch_features], dim=0)
        
        # object features
        start_time = time.time()
        obj_3D_embeddings = self.model.forward_scene_graph(data_dict)  # (O, C*)
        obj_3D_embeddings = self.model.obj_embedding_encoder(obj_3D_embeddings) # (O, C*)
        obj_3D_embeddings_norm = F.normalize(obj_3D_embeddings, dim=-1)
        scenegraph_emb_time = time.time() - start_time
        num_objs = obj_3D_embeddings.shape[0]
        num_scenes = len(data_dict['scene_graphs']['scene_ids']) 
        return patch_features_batch, obj_3D_embeddings_norm, forward_time, \
            scenegraph_emb_time, num_objs, num_scenes
    
    def model_forward_with_GlobalDescriptor(self, data_dict):
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
        
        # global descriptor
        patch_global_descriptor, obj_global_descriptors = None, None
        if self.use_global_descriptor:
            patch_features_batch_norm = F.normalize(patch_features_batch, dim=1)
            patch_global_descriptor, obj_global_descriptors = \
                self.model.forward_global_descriptor(patch_features_batch_norm, obj_3D_embeddings_norm, data_dict)

        return patch_features_batch, obj_3D_embeddings_norm, forward_time, patch_global_descriptor, obj_global_descriptors

    
    def room_retrieval_dict(self, data_dict, dataset, room_retrieval_record, record_retrieval = False):
        
        # room retrieval with scan point cloud
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_S".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_S".format(k): 0. for k in top_k_list}
        top_k_recall_global = {"R@{}_T_G".format(k): 0. for k in top_k_list}
        top_k_recall_global_non_temporal = {"R@{}_NT_G".format(k): 0. for k in top_k_list}
        
        retrieval_time_temporal = 0.
        retrieval_time_non_temporal = 0.
        img_forward_time = 0.
        matched_obj_idxs, matched_obj_idxs_temp = None, None
        
        obj_ids = data_dict['scene_graphs']['obj_ids']
        obj_ids_cpu = torch_util.release_cuda_torch(obj_ids)
        
        # get embeddings
        if self.use_global_descriptor:
            patch_features_batch, obj_3D_embeddings_norm, forward_time, patch_global_descriptor, obj_global_descriptors = \
                self.model_forward_with_GlobalDescriptor(data_dict)
            patch_global_descriptor_norm = F.normalize(patch_global_descriptor, dim=1)
        else:
            patch_features_batch, obj_3D_embeddings_norm, forward_time, \
                scenegraph_emb_time, num_objs, num_scenes = \
                    self.model_forward(data_dict)
        for batch_i in range(batch_size):
            patch_features = patch_features_batch[batch_i]
            # non-temporal
            room_score_scans_NT = {}
            assoc_data_dict = data_dict['assoc_data_dict'][batch_i]
            candidates_obj_sg_idxs = assoc_data_dict['scans_sg_obj_idxs']
            candata_scan_obj_idxs = assoc_data_dict['candata_scan_obj_idxs']
            cadidate_scans_semantic_ids = assoc_data_dict['cadidate_scans_semantic_ids']
            if self.use_tf_idf:
                # reweight_matrix_scans = assoc_data_dict['reweight_matrix_scans']
                # reweight_matrix_scans = torch_util.release_cuda_torch(reweight_matrix_scans)
                
                """
                nid -> number of patched of class that belongs to object i in image d
                nd -> number of patches in image d
                N -> total number of rooms
                ni -> number of rooms that contains object i
                """
                n_scenes_per_sem = assoc_data_dict['n_scenes_per_sem'] # ni
                n_scenes = len(candata_scan_obj_idxs) #  N
                
            target_scan_id = data_dict['scan_ids'][batch_i]
            candidates_objs_embeds_scan = obj_3D_embeddings_norm[candidates_obj_sg_idxs]
            ## start room retrieval in cpu
            patch_features_cpu = patch_features.cpu()
            candata_scan_obj_idxs_cpu = torch_util.release_cuda_torch(candata_scan_obj_idxs)
            obj_3D_embeddings_norm_cpu_scan = torch_util.release_cuda_torch(candidates_objs_embeds_scan)
            cadidate_scans_semantic_ids = torch_util.release_cuda_torch(cadidate_scans_semantic_ids)
            candidates_obj_embeds = {
                candidate_scan_id: obj_3D_embeddings_norm_cpu_scan[candata_scan_obj_idxs_cpu[candidate_scan_id]] \
                for candidate_scan_id in candata_scan_obj_idxs_cpu}
            start_time = time.time()
            patch_features_cpu_norm = F.normalize(patch_features_cpu, dim=1)
            for candidate_scan_id in candidates_obj_embeds:
                candidate_obj_embeds = candidates_obj_embeds[candidate_scan_id]
                patch_obj_sim = patch_features_cpu_norm@candidate_obj_embeds.T
                if self.use_tf_idf:
                    # reweight_obj_matrixs = reweight_matrix_scans[candidate_scan_id] + 0.5
                    # matched_candidate_objs_idxs = patch_obj_sim.argmax(dim=1)
                    # matched_sim = patch_obj_sim.gather(1, matched_candidate_objs_idxs.unsqueeze(1)).squeeze(1)
                    # reweight_patch_obj_sim = matched_sim * reweight_obj_matrixs[matched_candidate_objs_idxs]
                    # score = reweight_patch_obj_sim.sum().item() / reweight_obj_matrixs.sum().item()
                    
                    matched_candidate_objs_idxs = patch_obj_sim.argmax(dim=1)
                    matched_sim = patch_obj_sim.gather(1, matched_candidate_objs_idxs.unsqueeze(1)).squeeze(1)
                    
                    # get semantic category of matched objects
                    matched_obj_sem_ids = cadidate_scans_semantic_ids[candata_scan_obj_idxs_cpu[candidate_scan_id]][matched_candidate_objs_idxs]
                    unique_sem_ids, inverse_indices, counts = torch.unique(
                        matched_obj_sem_ids, return_counts=True, return_inverse=True)
                    reweight_matrix_uniq = torch.zeros_like(unique_sem_ids, dtype=torch.float32)
                    for idx, (sem_id, count) in enumerate(zip(unique_sem_ids, counts)):
                        N = torch.tensor(n_scenes, dtype=torch.float32)
                        ni = torch.tensor(len(n_scenes_per_sem[sem_id.item()]), dtype=torch.float32)
                        nid = torch.tensor( count, dtype=torch.float32)
                        nd = torch.tensor(matched_sim.shape[0], dtype=torch.float32)
                        # tf-idf
                        reweight_matrix_uniq[idx] = nid * (1 + torch.log(N/ni)) / nd
                    reweight_matrix = reweight_matrix_uniq[inverse_indices]
                    matched_candidate_obj_sim = matched_sim * reweight_matrix
                    score = matched_candidate_obj_sim.sum().item() / reweight_matrix.sum().item()
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
            matched_obj_ids = obj_ids_cpu_scan[matched_obj_idxs]
            matched_obj_cates = cadidate_scans_semantic_ids[candata_scan_obj_idxs_cpu[target_scan_id]][matched_obj_idxs]
            ## calculate matched object ids between patch and all candidate rooms 
            matched_obj_idxs_allscans = (patch_features_cpu_norm @ obj_3D_embeddings_norm_cpu_scan.T).argmax(dim=1)
            matched_obj_cates_allscans = cadidate_scans_semantic_ids[matched_obj_idxs_allscans]
            ### is correct for patch and object match of all candidate rooms 
            e1i_matrix = assoc_data_dict['e1i_matrix'].cpu().numpy()
            is_patch_correct_allscans = \
                np.take_along_axis(e1i_matrix, matched_obj_idxs_allscans.reshape(-1,1).cpu().numpy(), axis=1).reshape(-1)
            
            
            if self.use_global_descriptor:
                patch_global_descriptor_pb = patch_global_descriptor_norm[batch_i:batch_i+1]
                room_score_global_NT = {}
                for candidate_scan_id in candata_scan_obj_idxs.keys():
                    candidate_global_descriptor = obj_global_descriptors[candidate_scan_id]
                    candidate_global_descriptor_norm = F.normalize(candidate_global_descriptor, dim=1)
                    room_score_global_NT[candidate_scan_id] = patch_global_descriptor_pb@candidate_global_descriptor_norm.T
                room_sorted_global_scores_NT =  [item[0] for item in 
                                                    sorted(room_score_global_NT.items(), key=lambda x: x[1], reverse=True)]
                for k in top_k_list:
                    if target_scan_id in room_sorted_global_scores_NT[:k]:
                        top_k_recall_global_non_temporal["R@{}_NT_G".format(k)] += 1
                
            # temporal
            room_score_scans_T = {}
            assoc_data_dict_temp = data_dict['assoc_data_dict_temp'][batch_i]
            candidates_obj_sg_idxs = assoc_data_dict_temp['scans_sg_obj_idxs']
            candata_scan_obj_idxs = assoc_data_dict_temp['candata_scan_obj_idxs']
            cadidate_scans_semantic_ids = assoc_data_dict_temp['cadidate_scans_semantic_ids']
            if self.use_tf_idf:
                # reweight_matrix_scans = assoc_data_dict_temp['reweight_matrix_scans']
                # reweight_matrix_scans = torch_util.release_cuda_torch(reweight_matrix_scans)
                """
                nid -> number of patched of class that belongs to object i in image d
                nd -> number of patches in image d
                N -> total number of rooms
                ni -> number of rooms that contains object i
                """
                n_scenes_per_sem = assoc_data_dict_temp['n_scenes_per_sem'] # ni
                n_scenes = len(candata_scan_obj_idxs) #  N
            target_scan_id = data_dict['scan_ids_temp'][batch_i]
            candidates_objs_embeds_scan = obj_3D_embeddings_norm[candidates_obj_sg_idxs]
            ## start room retrieval in cpu
            candata_scan_obj_idxs_cpu = torch_util.release_cuda_torch(candata_scan_obj_idxs)
            obj_3D_embeddings_norm_cpu_scan = torch_util.release_cuda_torch(candidates_objs_embeds_scan)
            cadidate_scans_semantic_ids = torch_util.release_cuda_torch(cadidate_scans_semantic_ids)
            candidates_obj_embeds = {
                candidate_scan_id: obj_3D_embeddings_norm_cpu_scan[candata_scan_obj_idxs_cpu[candidate_scan_id]] \
                for candidate_scan_id in candata_scan_obj_idxs_cpu}
            start_time = time.time()
            for candidate_scan_id in candidates_obj_embeds:
                candidate_obj_embeds = candidates_obj_embeds[candidate_scan_id]
                patch_obj_sim = patch_features_cpu_norm@candidate_obj_embeds.T
                if self.use_tf_idf:
                    # reweight_obj_matrixs = reweight_matrix_scans[candidate_scan_id]
                    # matched_candidate_objs_idxs = patch_obj_sim.argmax(dim=1)
                    # matched_sim = patch_obj_sim.gather(1, matched_candidate_objs_idxs.unsqueeze(1)).squeeze(1)
                    # reweight_patch_obj_sim = matched_sim * reweight_obj_matrixs[matched_candidate_objs_idxs]
                    # score = reweight_patch_obj_sim.sum().item() / reweight_obj_matrixs.sum().item()
                    matched_candidate_objs_idxs = patch_obj_sim.argmax(dim=1)
                    matched_sim = patch_obj_sim.gather(1, matched_candidate_objs_idxs.unsqueeze(1)).squeeze(1)
                    # get semantic category of matched objects
                    matched_obj_sem_ids = cadidate_scans_semantic_ids[candata_scan_obj_idxs_cpu[candidate_scan_id]][matched_candidate_objs_idxs]
                    unique_sem_ids, inverse_indices, counts = torch.unique(
                        matched_obj_sem_ids, return_counts=True, return_inverse=True)
                    reweight_matrix_uniq = torch.zeros_like(unique_sem_ids, dtype=torch.float32)
                    for idx, (sem_id, count) in enumerate(zip(unique_sem_ids, counts)):
                        N = torch.tensor(n_scenes, dtype=torch.float32)
                        ni = torch.tensor(len(n_scenes_per_sem[sem_id.item()]), dtype=torch.float32)
                        nid = torch.tensor( count, dtype=torch.float32)
                        nd = torch.tensor(matched_sim.shape[0], dtype=torch.float32)
                        # tf-idf
                        reweight_matrix_uniq[idx] = nid * (1 + torch.log(N/ni)) / nd
                    reweight_matrix = reweight_matrix_uniq[inverse_indices]
                    matched_candidate_obj_sim = matched_sim * reweight_matrix
                    score = matched_candidate_obj_sim.sum().item() / reweight_matrix.sum().item()
                else:
                    matched_candidate_obj_sim = torch.max(patch_obj_sim, dim=1)[0]
                    score = matched_candidate_obj_sim.sum().item()
                room_score_scans_T[candidate_scan_id] = score

            room_sorted_by_scores = [item[0] for item in sorted(room_score_scans_T.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if target_scan_id in room_sorted_by_scores[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            retrieval_time_temporal += time.time() - start_time
            
            ## calculate matched object ids between patch and the target room
            matched_obj_idxs_temp = (patch_features_cpu_norm @ candidates_obj_embeds[target_scan_id].T).argmax(dim=1)
            obj_ids_cpu_scan = obj_ids_cpu[candidates_obj_sg_idxs.cpu()][candata_scan_obj_idxs_cpu[target_scan_id]]
            matched_obj_ids_temp = obj_ids_cpu_scan[matched_obj_idxs_temp]
            matched_obj_cates_temp = cadidate_scans_semantic_ids[candata_scan_obj_idxs_cpu[target_scan_id]][matched_obj_idxs_temp]
            ## calculate matched object ids between patch and all candidate rooms 
            matched_obj_idxs_allscans = (patch_features_cpu_norm @ obj_3D_embeddings_norm_cpu_scan.T).argmax(dim=1)
            matched_obj_cates_allscans_temp = cadidate_scans_semantic_ids[matched_obj_idxs_allscans]
            ### is correct for patch and object match of all candidate rooms 
            e1i_matrix_temp = assoc_data_dict_temp['e1i_matrix'].cpu().numpy()
            is_patch_correct_allscans_temp = \
                np.take_along_axis(e1i_matrix_temp, matched_obj_idxs_allscans.reshape(-1,1).cpu().numpy(), axis=1).reshape(-1)
            
            if self.use_global_descriptor:
                patch_global_descriptor_pb = patch_global_descriptor_norm[batch_i:batch_i+1]
                room_score_global_T = {}
                for candidate_scan_id in candata_scan_obj_idxs.keys():
                    candidate_global_descriptor = obj_global_descriptors[candidate_scan_id]
                    candidate_global_descriptor_norm = F.normalize(candidate_global_descriptor, dim=1)
                    room_score_global_T[candidate_scan_id] = patch_global_descriptor_pb@candidate_global_descriptor_norm.T
                room_sorted_global_scores_T =  [item[0] for item in 
                                                    sorted(room_score_global_T.items(), key=lambda x: x[1], reverse=True)]
                for k in top_k_list:
                    if target_scan_id in room_sorted_global_scores_T[:k]:
                        top_k_recall_global["R@{}_T_G".format(k)] += 1
            
            ## gt
            gt_obj_ids = data_dict['obj_2D_patch_anno_flatten_list'][batch_i].cpu().numpy()
            gt_obj_cates = assoc_data_dict['gt_patch_cates']
            gt_obj_cates_temp = assoc_data_dict_temp['gt_patch_cates']
            
            # retrieva_record
            scan_id = data_dict['scan_ids'][batch_i]
            if scan_id not in room_retrieval_record:
                room_retrieval_record[scan_id] = {'frames_retrieval': {}, 'sem_cat_id2name': dataset.obj_nyu40_id2name}
                room_retrieval_record[scan_id]['candidates_scan_ids'] = dataset.candidate_scans[scan_id]
                room_retrieval_record[scan_id]['obj_ids'] = dataset.scene_graphs[scan_id]['obj_ids']
            frame_idx = data_dict['frame_idxs'][batch_i]
            frame_retrieval = {
                'frame_idx': frame_idx,
                'temporal_scan_id': data_dict['scan_ids_temp'][batch_i],
                # non-temp
                ## target scan
                'matched_obj_ids': matched_obj_ids,
                'matched_obj_cates': matched_obj_cates,
                ## all scans
                'is_patch_correct_allscans': is_patch_correct_allscans,
                'matched_obj_cates_allscans': matched_obj_cates_allscans,
                # temp
                ## target scan
                'matched_obj_ids_temp': matched_obj_ids_temp,
                'matched_obj_cates_temp': matched_obj_cates_temp,
                ## all scans
                'is_patch_correct_allscans_temp': is_patch_correct_allscans_temp,
                'matched_obj_cates_allscans_temp': matched_obj_cates_allscans_temp,
                ## gt category
                'gt_anno': gt_obj_ids,
                'gt_obj_cates': gt_obj_cates,
                'gt_obj_cates_temp': gt_obj_cates_temp,
                ## retrieval scores
                'room_score_scans_NT': room_score_scans_NT,
                'room_score_scans_T': room_score_scans_T,
            }
            room_retrieval_record[scan_id]['frames_retrieval'][frame_idx] = frame_retrieval

        # average over batch
        for k in top_k_list:
            top_k_recall_temporal["R@{}_T_S".format(k)] /= 1.0*batch_size
            top_k_recall_non_temporal["R@{}_NT_S".format(k)] /= 1.0*batch_size
            top_k_recall_global["R@{}_T_G".format(k)] /= 1.0*batch_size
            top_k_recall_global_non_temporal["R@{}_NT_G".format(k)] /= 1.0*batch_size
            
        retrieval_time_temporal = retrieval_time_temporal / (1.0*batch_size)
        retrieval_time_non_temporal = retrieval_time_non_temporal / (1.0*batch_size)
        img_forward_time = forward_time / (1.0*batch_size)
        
        result = {
            'img_forward_time': img_forward_time,
            'time_T_S': retrieval_time_temporal,
            'time_NT_S': retrieval_time_non_temporal,
            'scenegraph_emb_time_per_scene': scenegraph_emb_time / (1.0*num_scenes),
            'scenegraph_emb_time_per_obj': scenegraph_emb_time / (1.0*num_objs),
        }
        result.update(top_k_recall_temporal)
        result.update(top_k_recall_non_temporal)
        if self.use_global_descriptor:
            result.update(top_k_recall_global)
            result.update(top_k_recall_global_non_temporal)
        
        return result

    def room_retrieval_val(self):
        # val 
        with torch.no_grad():
            data_dicts = tqdm.tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader))
            for iteration, data_dict in data_dicts:
                data_dict = torch_util.to_cuda(data_dict)
                result = self.room_retrieval_dict(data_dict, self.val_dataset, self.val_room_retrieval_record, True)
                self.val_room_retrieval_summary.update_from_result_dict(result)
                torch.cuda.empty_cache()
        val_items = self.val_room_retrieval_summary.tostringlist()
        # write metric to file
        val_file = osp.join(self.output_dir, 'val_result.txt')
        common.write_to_txt(val_file, val_items)
        # write retrieval record to file
        retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_val.pkl')
        common.write_pkl_data(self.val_room_retrieval_record, retrieval_record_file)
        ## statistics analysis
        val_retrieval_statistics = RetrievalStatistics(
            retrieval_records_dir = self.output_dir,  
            retrieval_records = self.val_room_retrieval_record, 
                split= 'val')
        val_retrieval_statistics.generateStaistics()
        
        # # test 
        # with torch.no_grad():
        #     data_dicts = tqdm.tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader))
        #     for iteration, data_dict in data_dicts:
        #         data_dict = torch_util.to_cuda(data_dict)
        #         result = self.room_retrieval_dict(data_dict, self.test_dataset,self.test_room_retrieval_record, True)
        #         self.test_room_retrieval_summary.update_from_result_dict(result)
        #         torch.cuda.empty_cache()
        # test_items = self.test_room_retrieval_summary.tostringlist()
        # # write metric to file
        # test_file = osp.join(self.output_dir, 'test_result.txt')
        # common.write_to_txt(test_file, test_items)
        # # write retrieval record to file
        # retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_test.pkl')
        # common.write_pkl_data(self.test_room_retrieval_record, retrieval_record_file)
            
        # ## statistics analysis
        # test_retrieval_statistics = RetrievalStatistics(
        #     retrieval_records_dir = self.output_dir,  retrieval_records = self.test_room_retrieval_record,
        #         split= 'test')
        # test_retrieval_statistics.generateStaistics()
            

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