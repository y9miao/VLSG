import os, sys
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from GCVit.models import gc_vit

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

class PatchObjectAligner(nn.Module):
    def __init__(self, 
                 backbone,
                 num_reduce,
                 backbone_dim,
                 img_transpose, 
                 patch_hidden_dims,
                 patch_encoder_dim,
                 obj_embedding_dim,
                 obj_embedding_hidden_dims,
                 obj_encoder_dim,
                 drop
                 ):
        super().__init__()
        
        # backbone 
        self.backbone = backbone
        reduce_list = [gc_vit.ReduceSize(dim=backbone_dim, keep_dim=True)
                            for i in range(num_reduce)]
        self.reduce_layers = nn.Sequential(*reduce_list)
        
        # patch feature encoder
        self.img_transpose = img_transpose
        self.patch_encoder = Mlps(backbone_dim, hidden_features = patch_hidden_dims, 
                                 out_features= patch_encoder_dim, drop = drop)
        
        # obj embedding encoder
        self.obj_embedding_encoder = Mlps(obj_embedding_dim, hidden_features = obj_embedding_hidden_dims, 
                                        out_features = obj_encoder_dim, drop = drop)
        
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
        patch_features = self.reduce_layers(features) # (B, H/32, W/32, C'); input channel last,output channel last 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*); input channel last,output channel last 
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        
        # obj encoding for each batch
        batch_size = data_dict['batch_size']
        obj_3D_features_list = []
        patch_obj_sim_list = []
        patch_patch_sim_list = []
        for batch_i in range(batch_size):
            # get patch and obj features
            obj_embeddings = data_dict['obj_3D_embeddings_list'][batch_i] 
            # # (O, C*); input channel last,output channel last
            obj_features = self.obj_embedding_encoder(obj_embeddings) 
            obj_3D_features_list.append(obj_features)
            
            # calculate similarity between patches and objs
            # patch features per batch
            patch_features_pb = patch_features[batch_i] # (P_H*P_W, C*)
            patch_features_pb_norm = F.normalize(patch_features_pb, dim=-1)
            obj_features_pb = obj_features
            obj_features_pb_norm = F.normalize(obj_features_pb, dim=-1)
            # calculate patch-object similarity (P_H*P_W, O)
            patch_obj_sim = torch.mm(patch_features_pb_norm, obj_features_pb_norm.permute(1, 0))
            patch_obj_sim_list.append(patch_obj_sim)
            # calculate patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = torch.mm(patch_features_pb_norm, patch_features_pb_norm.permute(1, 0))
            patch_patch_sim_list.append(patch_patch_sim)
        
        embs = {}
        embs['patch_features'] = patch_features # (B, P_H*P_W, C*)
        embs['obj_features'] = obj_3D_features_list # B - [O, C*]
        embs['patch_obj_sim'] = patch_obj_sim_list # B - [P_H*P_W, O]
        embs['patch_patch_sim'] = patch_patch_sim_list # B - [P_H*P_W, P_H*P_W]
        
        return embs

    