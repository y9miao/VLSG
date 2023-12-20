import torch
from torch import nn
import torch.nn.functional as F

class ICLLoss(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(ICLLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, embs, data_dict):
        patch_features = embs['patch_features'] # (B, P_H, P_W, C*)
        patch_features = patch_features.flatten(1, 2) # (B, P_H*P_W, C*)
        obj_features = embs['obj_features']  # # B - [O, C*]

        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # patch features per batch
            patch_features_pb = patch_features[batch_i] # (P_H, P_W, C*)
            patch_features_pb_norm = F.normalize(patch_features_pb, dim=-1)
            # obj_features per batch
            obj_features_pb = obj_features[batch_i] # (O, C*)
            obj_features_pb_norm = F.normalize(obj_features_pb, dim=-1)
            # calculate patch-object similarity (P_H*P_W, O)
            patch_obj_sim = torch.mm(patch_features_pb_norm, obj_features_pb_norm.permute(1, 0))
            patch_obj_sim_exp = torch.exp(patch_obj_sim / self.temp)
            # calculate patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = torch.mm(patch_features_pb_norm, patch_features_pb_norm.permute(1, 0))
            patch_patch_sim_exp = torch.exp(patch_patch_sim / self.temp)
            
            # calculate loss
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            delta_E1i_E2i = (e1i_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
            sum_delta_E1i_E1j = (e1j_matrix * patch_patch_sim_exp).sum(dim=-1) # (N_P)
            sum_delta_E1i_E2j = (e2j_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
            loss_patch_side = delta_E1i_E2i / (delta_E1i_E2i + sum_delta_E1i_E1j + sum_delta_E1i_E2j + self.epsilon)
            
            # mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            
            if e1i_valid.any():
                loss_patch_side = loss_patch_side[e1i_valid]
                loss = -torch.log(loss_patch_side + self.epsilon)
                loss_batch = loss if loss_batch is None else torch.cat((loss_batch, loss), dim=0)
        
                # calculate match success ratio
                matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
                matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs) # (N_P, 1)
                matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
                matched_obj_labels = matched_obj_labels[e1i_valid]
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
        
        if loss_batch is not None:
            return {
                'loss': loss_batch.mean(),
                'matched_success_ratio': matched_success_batch.float().mean()
            }
        else:
            return {
                'loss': 0.,
                'matched_success_ratio': 0.
            }