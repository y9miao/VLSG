# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mainly copy-paste from https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# sgaliner modules 
from models.sgaligner.src.aligner.networks.gat import MultiGAT
from models.sgaligner.src.aligner.networks.pointnet import PointNetfeat
from models.sgaligner.src.aligner.networks.pct import NaivePCT
# model utils
from model_utils import Attention, TransformerEncoderLayer, Mlps, PatchAggregator

    
class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb
    
class MultiModalEncoder(nn.Module):
    def __init__(self, modules, rel_dim, attr_dim, img_feat_dim,
                 hidden_units=[3, 128, 128], heads = [2, 2], emb_dim = 100, pt_out_dim = 256,
                 img_emb_dim = 256,
                 dropout = 0.0, attn_dropout = 0.0, instance_norm = False,
                 img_aggregation_mode="mean", use_pos_enc=False):
        super(MultiModalEncoder, self).__init__()
        self.modules = modules
        self.pt_out_dim = pt_out_dim
        self.rel_dim = rel_dim
        self.emb_dim = emb_dim
        self.img_emb_dim = img_emb_dim
        self.attr_dim =  attr_dim
        self.hidden_units = hidden_units
        self.heads = heads
        
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.instance_norm = instance_norm
        self.inner_view_num = len(self.modules) # Point Net + Structure Encoder + Meta Encoder
        self.use_pos_enc = use_pos_enc
        
        if 'point' in self.modules:
            self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=3, input_transform=False, feature_transform=False, out_size=self.pt_out_dim)
            self.object_embedding = nn.Linear(self.pt_out_dim, self.emb_dim)
        elif 'pct' in self.modules:
            self.object_encoder = NaivePCT()
            self.object_embedding = nn.Linear(self.pt_out_dim, self.emb_dim)
        # else:
        #     raise NotImplementedError
        if 'gat' in self.modules:
            self.structure_encoder = MultiGAT(n_units=self.hidden_units, n_heads=self.heads, dropout=self.dropout)
            self.structure_embedding = nn.Linear(256, self.emb_dim)
            
        if 'rel' in self.modules:
            self.meta_embedding_rel = nn.Linear(self.rel_dim, self.emb_dim)
        
        if 'attr' in self.modules:
            self.meta_embedding_attr = nn.Linear(self.attr_dim, self.emb_dim)
            
        if 'img_patch' in self.modules:
            img_encode_dim = 256
            self.img_patch_encoder = Mlps(in_features=img_feat_dim, 
                        hidden_features=[512], out_features=img_encode_dim)
            if self.use_pos_enc:
                self.pose_encoder = Mlps(in_features=7, hidden_features=[128], out_features=img_encode_dim)
            
            # how to aggregate multi-view image features
            self.img_aggregation_mode = img_aggregation_mode
            if self.img_aggregation_mode == "mean":
                pass
            elif self.img_aggregation_mode == "max":
                pass
            elif self.img_aggregation_mode == "transformer":
                self.use_transformer_aggregator = True
                # self.img_aggregator = \
                #     TransformerEncoderLayer(d_model=img_encode_dim, nhead=4)
                self.multiview_encoder = PatchAggregator(d_model=img_encode_dim, nhead=4, num_layers=1, dropout=self.dropout)
                self.multiview_norm = nn.LayerNorm(img_encode_dim)
            else:
                raise NotImplementedError
            self.img_patch_embedding = nn.Linear(img_encode_dim, self.img_emb_dim)
        
        self.fusion = MultiModalFusion(modal_num=self.inner_view_num, with_weight=1)
        
    def forward(self, data_dict):
        batch_size = data_dict['batch_size']
        embs = {}

        for module in self.modules:
            if module == 'gat':
                rel_pose = data_dict['tot_rel_pose'].float()
                structure_embed = None
                start_object_idx = 0
                start_edge_idx = 0
                for idx in range(batch_size):
                    object_count = data_dict['graph_per_obj_count'][idx][0]
                    edges_count = data_dict['graph_per_edge_count'][idx][0]
                    
                    objects_rel_pose = rel_pose[start_object_idx : start_object_idx + object_count]
                    start_object_idx += object_count
                    
                    edges = torch.transpose(data_dict['edges'][start_edge_idx : start_edge_idx + edges_count], 0, 1).to(torch.int32)
                    start_edge_idx += edges_count

                    structure_embedding = self.structure_encoder(objects_rel_pose, edges)
                    
                    structure_embed = torch.cat([structure_embedding]) if structure_embed is None else \
                                   torch.cat([structure_embed, structure_embedding]) 

                emb = self.structure_embedding(structure_embed)
            
            elif module in ['point', 'pct']:
                object_points = data_dict['tot_obj_pts'].permute(0, 2, 1)
                emb = self.object_encoder(object_points)
                emb = self.object_embedding(emb)

            elif module == 'rel':
                bow_vec_object_edge_feats = data_dict['tot_bow_vec_object_edge_feats'].float()
                emb = self.meta_embedding_rel(bow_vec_object_edge_feats)
            
            elif module == 'attr':
                bow_vec_object_attr_feats = data_dict['tot_bow_vec_object_attr_feats'].float() 
                emb = self.meta_embedding_attr(bow_vec_object_attr_feats)
                
            elif module == 'img_patch':
                obs_img_patch_emb = None
                start_object_idx = 0
                for idx in range(batch_size):
                    scan_id = data_dict['scene_ids'][idx][0]
                    obj_count = data_dict['graph_per_obj_count'][idx][0]
                    obj_ids = data_dict['obj_ids'][start_object_idx: start_object_idx + obj_count]
                    img_patch_feat_scan = data_dict['obj_img_patches'][scan_id]
                    for obj in obj_ids:
                        img_patches = img_patch_feat_scan[obj]
                        img_patches_encoded = self.img_patch_encoder(img_patches)
                        if self.use_pos_enc:
                            img_poses = data_dict['obj_img_poses'][scan_id][obj]
                            img_poses = img_poses.unsqueeze(0)
                            img_patches_encoded = img_patches_encoded + self.pose_encoder(img_poses)
                        
                        # aggregate multi-view image features
                        if self.img_aggregation_mode == "mean":
                            img_multiview_encode = torch.mean(img_patches_encoded, dim=0, keepdim=True)
                        elif self.img_aggregation_mode == "max":
                            img_multiview_encode, _ = torch.max(img_patches_encoded, dim=0, keepdim=True)
                        elif self.img_aggregation_mode == "transformer":
                            img_patches_attn = self.multiview_encoder(img_patches_encoded.unsqueeze(0))
                            img_patches_attn = self.multiview_norm(img_patches_attn)
                            img_patches_attn = img_patches_attn.squeeze(0)
                            img_multiview_encode, _ = torch.max(img_patches_encoded + img_patches_attn, dim=0, keepdim=True)
                        obs_img_patch_emb = torch.cat([img_multiview_encode]) if obs_img_patch_emb is None else \
                                    torch.cat([obs_img_patch_emb, img_multiview_encode])
                                    
                    start_object_idx += obj_count
                emb = self.img_patch_embedding(obs_img_patch_emb)
            else:
                raise NotImplementedError
            
            embs[module] = emb
        
        if len(self.modules) > 1:
            all_embs = []
            for module in self.modules:
                all_embs.append(embs[module])
            
            joint_emb = self.fusion(all_embs)
            embs['joint'] = joint_emb
        else:
            embs['joint'] = embs[self.modules[0]]
        
        return embs