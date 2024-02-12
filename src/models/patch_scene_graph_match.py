import os, sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)
src_dir = os.path.dirname(parent_dir)
sys.path.append(src_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
# 2D image feature extractor
from GCVit.models import gc_vit
# 3D scene graph feature extractor
from src.models.sg_encoder_compact import SceneGraphEncoder, Mlps


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

class PatchSceneGraphAligner(nn.Module):
    def __init__(self, 
                 backbone,
                 num_reduce,
                 backbone_dim,
                 patch_hidden_dims,
                 patch_encoder_dim,
                 scene_graph_modules,
                 scene_graph_in_dims,
                 scene_graph_encode_depth,
                 scene_graph_emb_dims,
                 gat_hidden_units,
                 gat_heads,
                 scene_graph_node_dim,
                 node_out_dim,
                 drop,
                 use_temporal,
                 multiview_transformer=False,):
        super().__init__()
        
        # backbone 
        self.backbone = backbone
        if num_reduce > 0:
            reduce_list = [gc_vit.ReduceSize(dim=backbone_dim, keep_dim=True)
                                for i in range(num_reduce)]
            self.reduce_layers = nn.Sequential(*reduce_list)
        else:
            self.reduce_layers = EncodeChannelSize(backbone_dim, backbone_dim, norm_layer=nn.LayerNorm)
        
        # patch feature encoder
        self.patch_encoder = Mlps(backbone_dim, hidden_features = patch_hidden_dims, 
                                 out_features= patch_encoder_dim, drop = drop)
        
        # 3D scene graph encoder
        scene_graph_in_dims_dict = {scene_graph_modules[i]: scene_graph_in_dims[i] \
            for i in range(len(scene_graph_modules))}
        scene_graph_emb_dims_dict = {scene_graph_modules[i]: scene_graph_emb_dims[i] \
            for i in range(len(scene_graph_modules))}
        scene_graph_encode_depth_dict = {scene_graph_modules[i]: scene_graph_encode_depth[i] \
            for i in range(len(scene_graph_modules))}
        self.sg_encoder = SceneGraphEncoder(scene_graph_modules, 
                                            in_dims = scene_graph_in_dims_dict,
                                            encode_depth = scene_graph_encode_depth_dict,
                                            encode_dims = scene_graph_emb_dims_dict,
                                            gat_hidden_units=gat_hidden_units,
                                            gat_heads = gat_heads,
                                            dropout = drop,
                                            use_transformer_aggregator=multiview_transformer)
        self.sg_node_dim_match = nn.Linear(scene_graph_node_dim, node_out_dim)
        
        # use temporal information
        self.use_temporal = use_temporal
        
    def forward(self, data_dict):
        # get data
        images = data_dict['images'] # (B, H, W, C)
        channel_last = True
        
        # patch encoding 
        images = _to_channel_first(images)
        channel_last = False
        features = self.backbone(images)[-1] # (B, C', H/32, W/32); input channel first,output channel first 
        features = _to_channel_last(features)
        channel_last = True
        patch_features = self.reduce_layers(features) # (B, H/64, W/64, C'); input channel last,output channel last 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*); input channel last,output channel last 
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        
        # sg encoding
        obj_3D_embeddings = self.forward_scene_graph(data_dict)  # (O, C*)
        obj_3D_embeddings = self.sg_node_dim_match(obj_3D_embeddings) # (O, C*)
        obj_3D_embeddings_norm = F.normalize(obj_3D_embeddings, dim=-1)
        ## as long as batch_size^2 < batch_size*num_candidates^2, it is faster to calculate similarity between objs and patches
        obj_3D_embeddings_sim = torch.mm(obj_3D_embeddings_norm, obj_3D_embeddings_norm.permute(1, 0)) # (O, O)
         
        # calculate similarity between patches and objs
        batch_size = data_dict['batch_size']
        
        patch_obj_sim_list = []
        patch_patch_sim_list = []
        obj_obj_sim_list = []
        patch_obj_sim_temp_list = []
        patch_patch_sim_temp_list = []
        obj_obj_sim_temp_list = []
        
        for batch_i in range(batch_size):
            # calculate similarity between patches and objs
            # patch features per batch
            patch_features_pb = patch_features[batch_i] # (P_H*P_W, C*)
            patch_features_pb_norm = F.normalize(patch_features_pb, dim=-1)
            assoc_data_dict = data_dict['assoc_data_dict'][batch_i]
            patch_obj_sim, patch_patch_sim, obj_obj_sim = self.calculate_similarity(
                obj_3D_embeddings_norm, patch_features_pb_norm, assoc_data_dict, obj_3D_embeddings_sim)
            patch_obj_sim_list.append(patch_obj_sim)
            patch_patch_sim_list.append(patch_patch_sim)
            obj_obj_sim_list.append(obj_obj_sim)
            # temporal information
            if self.use_temporal:
                assoc_data_dict = data_dict['assoc_data_dict_temp'][batch_i]
                patch_obj_sim_temp, patch_patch_sim_temp, obj_obj_sim_temp = self.calculate_similarity(
                    obj_3D_embeddings_norm, patch_features_pb_norm, assoc_data_dict, obj_3D_embeddings_sim)
                patch_obj_sim_temp_list.append(patch_obj_sim_temp)
                patch_patch_sim_temp_list.append(patch_patch_sim_temp)
                obj_obj_sim_temp_list.append(obj_obj_sim_temp)

        embs = {}
        embs['patch_raw_features'] = features # (B, H/32, W/32, C');
        embs['patch_features'] = patch_features # (B, P_H*P_W, C*)
        embs['obj_features'] = obj_3D_embeddings # [O, C*]
        embs['patch_obj_sim'] = patch_obj_sim_list # B - [P_H*P_W, N]
        embs['patch_patch_sim'] = patch_patch_sim_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim'] = obj_obj_sim_list # B - [N, N]
        embs['patch_obj_sim_temp'] = patch_obj_sim_temp_list # B - [P_H*P_W, N]
        embs['patch_patch_sim_temp'] = patch_patch_sim_temp_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim_temp'] = obj_obj_sim_temp_list
        return embs
    
    def calculate_similarity(self, obj_3D_embeddings_norm, patch_features_norm, 
                             assoc_data_dict, obj_3D_embeddings_sim):
        scan_objs_idx = assoc_data_dict['scans_sg_obj_idxs']
        scans_obj_embeddings_norm = obj_3D_embeddings_norm[scan_objs_idx, :] # (N, C*)
        patch_obj_sim = torch.mm(patch_features_norm, scans_obj_embeddings_norm.permute(1, 0)) # (P_H*P_W, N)
        patch_patch_sim = torch.mm(patch_features_norm, patch_features_norm.permute(1, 0)) # (P_H*P_W, P_H*P_W)
        obj_obj_sim = obj_3D_embeddings_sim[scan_objs_idx, :][:, scan_objs_idx] # (N, N)
        return patch_obj_sim, patch_patch_sim, obj_obj_sim
    
    def forward_with_patch_features(self, data_dict):
        # get data
        patch_features = data_dict['patch_features'] # (B, P_H*2, P_W*2, C*)
        
        # encoding patch features
        patch_features = self.reduce_layers(patch_features) 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*)
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        
        # sg encoding
        obj_3D_embeddings = self.forward_scene_graph(data_dict)  # (O, C*)
        obj_3D_embeddings = self.sg_node_dim_match(obj_3D_embeddings) # (O, C*)
        obj_3D_embeddings_norm = F.normalize(obj_3D_embeddings, dim=-1)
        ## as long as batch_size^2 < batch_size*num_candidates^2, it is faster to calculate similarity between objs and patches
        obj_3D_embeddings_sim = torch.mm(obj_3D_embeddings_norm, obj_3D_embeddings_norm.permute(1, 0)) # (O, O)
         
        # calculate similarity between patches and objs
        batch_size = data_dict['batch_size']
        
        patch_obj_sim_list = []
        patch_patch_sim_list = []
        obj_obj_sim_list = []
        patch_obj_sim_temp_list = []
        patch_patch_sim_temp_list = []
        obj_obj_sim_temp_list = []
        
        for batch_i in range(batch_size):
            # calculate similarity between patches and objs
            # patch features per batch
            patch_features_pb = patch_features[batch_i] # (P_H*P_W, C*)
            patch_features_pb_norm = F.normalize(patch_features_pb, dim=-1)
            assoc_data_dict = data_dict['assoc_data_dict'][batch_i]
            patch_obj_sim, patch_patch_sim, obj_obj_sim = self.calculate_similarity(
                obj_3D_embeddings_norm, patch_features_pb_norm, assoc_data_dict, obj_3D_embeddings_sim)
            patch_obj_sim_list.append(patch_obj_sim)
            patch_patch_sim_list.append(patch_patch_sim)
            obj_obj_sim_list.append(obj_obj_sim)
            # temporal information
            if self.use_temporal:
                assoc_data_dict = data_dict['assoc_data_dict_temp'][batch_i]
                patch_obj_sim_temp, patch_patch_sim_temp, obj_obj_sim_temp = self.calculate_similarity(
                    obj_3D_embeddings_norm, patch_features_pb_norm, assoc_data_dict, obj_3D_embeddings_sim)
                patch_obj_sim_temp_list.append(patch_obj_sim_temp)
                patch_patch_sim_temp_list.append(patch_patch_sim_temp)
                obj_obj_sim_temp_list.append(obj_obj_sim_temp)

        embs = {}
        embs['patch_features'] = patch_features # (B, P_H*P_W, C*)
        embs['obj_features'] = obj_3D_embeddings # [O, C*]
        embs['patch_obj_sim'] = patch_obj_sim_list # B - [P_H*P_W, N]
        embs['patch_patch_sim'] = patch_patch_sim_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim'] = obj_obj_sim_list # B - [N, N]
        embs['patch_obj_sim_temp'] = patch_obj_sim_temp_list # B - [P_H*P_W, N]
        embs['patch_patch_sim_temp'] = patch_patch_sim_temp_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim_temp'] = obj_obj_sim_temp_list
        return embs
    
    def forward2DImage(self, data_dict):
        # get data
        images = data_dict['images']
        
        # patch encoding 
        images = _to_channel_first(images)
        channel_last = False
        features = self.backbone(images)[-1] # (B, C', H/32, W/32); input channel first,output channel first 
        features = _to_channel_last(features)
        channel_last = True
        patch_features = self.reduce_layers(features) # (B, H/64, W/64, C'); input channel last,output channel last 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*); input channel last,output channel last 
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        
        return patch_features
    
    def forward_scene_graph(self, data_dict):
        scene_graph_dict = data_dict['scene_graphs']
        object_embeddings = self.sg_encoder(scene_graph_dict)
        return object_embeddings

class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EncodeChannelSize(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 norm_layer=nn.LayerNorm):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim_out, 1, 1, 0, bias=False),
        )
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        x = x + self.conv(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        return x