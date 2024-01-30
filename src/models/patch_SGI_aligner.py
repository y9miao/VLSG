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
from src.models.sg_encoder import MultiModalEncoder


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

class PatchSGIAligner(nn.Module):
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
                 sg_modules,
                 sg_rel_dim,
                 attr_dim,
                 img_feat_dim, 
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
        
        # 3D scene graph encoder
        self.sg_encoder = MultiModalEncoder(
            modules = sg_modules, rel_dim = sg_rel_dim, attr_dim=attr_dim, 
            img_feat_dim = img_feat_dim, dropout = drop)
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
        patch_features = self.reduce_layers(features) # (B, H/64, W/64, C'); input channel last,output channel last 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*); input channel last,output channel last 
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        
        # sg encoding
        obj_3D_embeddings_list = self.forward_scene_graph(data_dict)
        
        # obj encoding for each batch
        batch_size = data_dict['batch_size']
        obj_3D_features_list = []
        patch_obj_sim_list = []
        patch_patch_sim_list = []
        obj_obj_sim_list = []
        for batch_i in range(batch_size):
            # get patch and obj features
            obj_embeddings = obj_3D_embeddings_list[batch_i] 
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
            # calculate obj-obj similarity (O, O)
            obj_obj_sim = torch.mm(obj_features_pb_norm, obj_features_pb_norm.permute(1, 0))
            obj_obj_sim_list.append(obj_obj_sim)
        embs = {}
        embs['patch_raw_features'] = features # (B, H/32, W/32, C');
        embs['patch_features'] = patch_features # (B, P_H*P_W, C*)
        embs['obj_features'] = obj_3D_features_list # B - [O, C*]
        embs['patch_obj_sim'] = patch_obj_sim_list # B - [P_H*P_W, O]
        embs['patch_patch_sim'] = patch_patch_sim_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim'] = obj_obj_sim_list # B - [O, O]
        
        return embs
    
    def forward_with_patch_features(self, data_dict):
        # get data
        patch_features = data_dict['patch_features'] # (B, P_H*2, P_W*2, C*)
        
        # encoding patch features
        patch_features = self.reduce_layers(patch_features) 
        patch_features = self.patch_encoder(patch_features) # (B, P_H, P_W, C*)
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)

        # obj encoding for each batch
        batch_size = data_dict['batch_size']
        obj_3D_features_list = []
        patch_obj_sim_list = []
        patch_patch_sim_list = []
        obj_obj_sim_list = []
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
            # calculate obj-obj similarity (O, O)
            obj_obj_sim = torch.mm(obj_features_pb_norm, obj_features_pb_norm.permute(1, 0))
            obj_obj_sim_list.append(obj_obj_sim)
        embs = {}
        embs['patch_features'] = patch_features # (B, P_H*P_W, C*)
        embs['obj_features'] = obj_3D_features_list # B - [O, C*]
        embs['patch_obj_sim'] = patch_obj_sim_list # B - [P_H*P_W, O]
        embs['patch_patch_sim'] = patch_patch_sim_list # B - [P_H*P_W, P_H*P_W]
        embs['obj_obj_sim'] = obj_obj_sim_list # B - [O, O]
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
        # given scene graph info, get object embeddings
        object_embeddings_list = []
        
        for batch_i in range(data_dict['batch_size']):
            obj_embs_scans_pb = {}
            # inference
            scene_graph_dict = data_dict['scene_graphs_list'][batch_i]
            sg_obj_embs = self.sg_encoder(scene_graph_dict)
            # get object embeddings
            objs_sg_idx = 0
            scans_ids_pb = scene_graph_dict['scene_ids']
            for scan_idx, scan_id in enumerate(scans_ids_pb):
                scan_id = scan_id[0]
                obj_embs_scans_pb[scan_id] = {}
                ## get each obj emb
                obj_count = scene_graph_dict['tot_obj_count'][scan_idx]
                for obj_sg_idx in range(objs_sg_idx, objs_sg_idx+obj_count):
                    obj_id = scene_graph_dict['obj_ids'][obj_sg_idx]
                    obj_embs_scans_pb[scan_id][obj_id] = sg_obj_embs['joint'][obj_sg_idx]
                objs_sg_idx += obj_count
            # aggregate embeddings
            object_embeddings_pb = []
            obj_3D_idx2info = data_dict['obj_3D_idx2info_list'][batch_i]
            for obj_3D_idx in range(len(obj_3D_idx2info)):
                scan_id = obj_3D_idx2info[obj_3D_idx][0]
                obj_id = obj_3D_idx2info[obj_3D_idx][1]
                object_embeddings_pb.append(obj_embs_scans_pb[scan_id][obj_id])
            object_embeddings_pb = torch.stack(object_embeddings_pb, dim=0)
            
            # append to list
            object_embeddings_list.append(object_embeddings_pb)
            
        return object_embeddings_list