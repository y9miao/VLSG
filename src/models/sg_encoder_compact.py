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
from math import e
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# sgaliner modules 
from models.sgaligner.src.aligner.networks.pointnet import PointNetfeat
from torch_geometric.nn import GATConv, GCNConv

# model utils
from model_utils import Attention, TransformerEncoderLayer

class PatchAggregator(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        # transformer encoders
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # Initialize the [CLS] token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.transformer_encoder = TransformerEncoderLayer(d_model=d_model, d_model_inner=int(d_model/2), nhead=nhead)

    def forward(self, x):
        # # x: (batch_size, num_patches, d_model)
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # output = self.transformer_encoder(x)
        # cls_token_output = output[:, 0, :]
        # return cls_token_output
        
        # x: (batch_size, num_patches, d_model)
        output = self.transformer_encoder(x)
        return output
    
class Mlps(nn.Module):
    def __init__(self, in_features, hidden_features=[], out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.direct_output = False

        layers_list = []
        if(len(hidden_features) <= 0):
            assert out_features == in_features
            self.direct_output  = True
            # layers_list.append(nn.Linear(in_features, out_features))
            # layers_list.append(nn.Dropout(drop))
            
        else:
            hidden_features = [in_features] + hidden_features
            for i in range(len(hidden_features)-1):
                layers_list.append(nn.Linear(hidden_features[i], hidden_features[i+1]))
                layers_list.append(act_layer())
                layers_list.append(nn.Dropout(drop))
            layers_list.append(nn.Linear(hidden_features[-1], out_features))
            layers_list.append(nn.Dropout(drop))
            
        self.mlp_layers =  nn.Sequential(*layers_list)

    def forward(self, x):
        if self.direct_output:
            return x
        else:
            return self.mlp_layers(x)
    
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
    
class MultiGAT(nn.Module):
    def __init__(self, n_units, n_gat_heads=[2, 2], dropout=0.0, use_edge_feat = False):
        super(MultiGAT, self).__init__()
        
        self.use_edge_feat = use_edge_feat
        self.num_layers = len(n_units)
        self.dropout = dropout
        
        layer_stack = []
        # in_channels, out_channels, gat_heads
        for i in range(self.num_layers-1):
            in_channel = n_units[i] * n_gat_heads[i-1] if i else n_units[i]
            out_channel = n_units[i+1]
            layer_stack.append(GATConv(in_channels=in_channel, out_channels=out_channel, 
                                       cached=False, heads=n_gat_heads[i]))
        self.layer_stack = nn.ModuleList(layer_stack)
        
    def forward(self, x, edges):
        # TODO add support for edge features
        for idx, gat_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, edges)
            if idx+1 < self.num_layers:
                x = F.elu(x)
        return x
    
class SceneGraphEncoder(nn.Module):
    def __init__(self, modules, 
                 in_dims = {'point': 3, 'attr': 164, 'img_patch': 1536, 'rel': 41, 'gat': 3},
                 encode_depth = {'point':[256, 256], 'attr': [128,128], 'img_patch': [512, 256], 'rel': [128, 128] },
                 encode_dims = {'point': 128, 'attr': 128, 'img_patch': 256, 'rel': 128, 'gat': 128},
                 gat_hidden_units=[128, 128], gat_heads = [2, 2],
                 dropout = 0.0,  use_transformer_aggregator=False,
                 multi_modal_fusion = True):
        super(SceneGraphEncoder, self).__init__()
        self.modules = modules
        self.in_dims = in_dims
        self.encode_depth = encode_depth
        self.encode_dims = encode_dims
        self.gat_hidden_units = gat_hidden_units
        self.gat_heads = gat_heads
        
        self.dropout = dropout
        
        # multi modal encoder models for nodes
        self.context_num = 0
        if 'point' in self.modules:
            point_dim_in = self.in_dims['point']
            geometry_dim = 256
            point_encode_dim = self.encode_dims['point']
            
            self.geometry_encode = PointNetfeat(global_feat=True, batch_norm=True, point_size=point_dim_in, 
                        input_transform=False, feature_transform=False, out_size=geometry_dim)
            self.geometry_dim_encoder = Mlps(in_features=geometry_dim, hidden_features=encode_depth['point'], out_features=point_encode_dim)
        
        if 'attr' in self.modules:
            attr_dim_in = self.in_dims['attr']
            attr_encode_dim = self.encode_dims['attr']
            self.attr_encoder = Mlps(in_features=attr_dim_in, hidden_features=encode_depth['attr'], out_features=attr_encode_dim)
            
        if 'rel' in self.modules:
            self.rel_encoder = Mlps(in_features=self.in_dims['rel'], hidden_features=encode_depth['rel'], 
                                    out_features=self.encode_dims['rel'])
        
        if 'img_patch' in self.modules:
            # whether to use transformer_encoder
            self.use_transformer_aggregator = use_transformer_aggregator
            img_feat_input_dim = self.in_dims['img_patch']
            img_feat_encode_dim = self.encode_dims['img_patch']
            self.img_patch_encoder = Mlps(in_features=img_feat_input_dim, 
                        hidden_features=encode_depth['img_patch'], out_features=img_feat_encode_dim)
            self.multiview_encoder = PatchAggregator(d_model=img_feat_encode_dim, nhead=2, num_layers=1, dropout=self.dropout)
            self.multiview_norm = nn.LayerNorm(img_feat_encode_dim)
        
        if 'gat' in self.modules:
            self.structure_encoder = MultiGAT(n_units=[self.in_dims['gat']] +self.gat_hidden_units, 
                                              n_gat_heads=self.gat_heads, dropout=self.dropout)
            gat_out_dim = self.gat_hidden_units[-1] * self.gat_heads[-1]
            self.structure_embedding = nn.Linear(gat_out_dim, self.encode_dims['gat'])
        
        # graph attention network
        # node_feat_dim = sum([self.encode_dims[module] for module in self.modules])
        # gate_gat_hidden_units = [node_feat_dim] + gat_hidden_units
        # self.gate = MultiGAT(gate_gat_hidden_units, n_gat_heads=gat_heads, dropout=dropout)
        
        # multi modal fusion
        self.use_multi_modal_fusion = multi_modal_fusion
        self.multi_modal_fusion = MultiModalFusion(len(self.modules))
            
        
    def forward(self, data_dict):
        batch_size = data_dict['batch_size']
        
        object_points = data_dict['tot_obj_pts'].permute(0, 2, 1)
        bow_vec_object_attr_feats = data_dict['tot_bow_vec_object_attr_feats'].float() 
        bow_vec_object_edge_feats = data_dict['tot_bow_vec_object_edge_feats'].float()
        rel_pose = data_dict['tot_rel_pose'].float()
        
        embs = {}
        for module in self.modules:
            # encode each module
            # point geometry encoder
            if module == 'point':
                geo_features = self.geometry_encode(object_points)
                geo_encode = self.geometry_dim_encoder(geo_features)
                embs[module] = geo_encode
                
            elif module == 'attr':
                attr_encode = self.attr_encoder(bow_vec_object_attr_feats)
                embs[module] = attr_encode
                
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
                        img_patch_encode = self.img_patch_encoder(img_patches)
                        
                        if self.use_transformer_aggregator:
                            img_patches_attn = self.multiview_encoder(img_patch_encode.unsqueeze(0))
                            img_patches_attn = self.multiview_norm(img_patches_attn)
                            img_patches_attn = img_patches_attn.squeeze(0)
                            img_multiview_encode = torch.mean(img_patch_encode + img_patches_attn, dim=0, keepdim=True)
                            img_multiview_encode = img_multiview_encode
                            obs_img_patch_emb = torch.cat([img_multiview_encode]) if obs_img_patch_emb is None else \
                                        torch.cat([obs_img_patch_emb, img_multiview_encode])
                        else:
                            obs_img_patch_emb = torch.cat([img_patch_encode]) if obs_img_patch_emb is None else \
                                        torch.cat([obs_img_patch_emb, img_patches])
                    start_object_idx += obj_count

                
                embs[module] = obs_img_patch_emb
                
            elif module == 'rel':
                rel_encode = self.rel_encoder(bow_vec_object_edge_feats)
                embs[module] = rel_encode
                
            elif module == 'gat':
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

                embs[module] = self.structure_embedding(structure_embed)
            else:
                raise NotImplementedError
        

        
        # fusion of multi-modal embeddings
        if self.use_multi_modal_fusion:
            joint_emb = self.multi_modal_fusion([embs[module] for module in self.modules])
            return joint_emb
        else:
            # concatenate all embeddings
            node_embs = None
            for module in self.modules:
                modular_embs = embs[module]
                node_embs = torch.cat([node_embs, modular_embs], dim=1) \
                    if node_embs is not None else modular_embs
            return node_embs
                
        # # formulate edges
        # cur_obj_num = 0
        # start_edge_idx = 0
        # edges = data_dict['edges']
        # for idx in range(batch_size):
        #     object_count = data_dict['graph_per_obj_count'][idx][0]
        #     edges_count = data_dict['graph_per_edge_count'][idx][0]
            
        #     edges[start_edge_idx : start_edge_idx + edges_count] += cur_obj_num
            
        #     cur_obj_num += object_count
        #     start_edge_idx += edges_count
        # edges = torch.transpose(edges, 0, 1).to(torch.int32)
        # # graph attention network
        # embs = self.gate(node_embs, edges)
        # return joint_emb