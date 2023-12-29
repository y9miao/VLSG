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
        obj_features = embs['obj_features']  # # B - [O, C*]
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]

        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        patch_obj_sim_batch = []
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # calculate patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            patch_obj_sim_exp = torch.exp(patch_obj_sim / self.temp)
            # calculate patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
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
                    
            # save exp similarity of patch-object and patch-patch
            patch_obj_sim_batch.append(patch_patch_sim_exp)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        
        return loss_dict
    
class ICLLossBothSides(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(ICLLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, embs, data_dict):
        patch_features = embs['patch_features'] # (B, P_H, P_W, C*)
        obj_features = embs['obj_features']  # # B - [O, C*]
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]

        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        
        patch_obj_sim_batch = []
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # calculate patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            patch_obj_sim_exp = torch.exp(patch_obj_sim / self.temp)
            # calculate patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            patch_patch_sim_exp = torch.exp(patch_patch_sim / self.temp)
            
            # calculate 2D-to-3D loss
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            delta_E1i_E2i = (e1i_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
            sum_delta_E1i_E1j = (e1j_matrix * patch_patch_sim_exp).sum(dim=-1) # (N_P)
            sum_delta_E1i_E2j = (e2j_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
            loss_patch_side = delta_E1i_E2i / (delta_E1i_E2i + sum_delta_E1i_E1j + sum_delta_E1i_E2j + self.epsilon)
            
            # calculate 3D-to-2D loss
            f1i = e1i_matrix.transpose(0,1) # (O, N_P)
            
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
                    
            # save exp similarity of patch-object and patch-patch
            patch_obj_sim_batch.append(patch_patch_sim_exp)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        
        return loss_dict
    
class RoomRetrivalLoss(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(RoomRetrivalLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, embs, data_dict):
        patch_features = embs['patch_features'] # (B, P_H, P_W, C*)
        obj_features = embs['obj_features']  # # B - [O, C*]
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]

        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        patch_obj_sim_batch = []
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # calculate patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            patch_obj_sim_exp = torch.exp(patch_obj_sim / self.temp)
            # calculate patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
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
                    
            # save exp similarity of patch-object and patch-patch
            patch_obj_sim_batch.append(patch_patch_sim_exp)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        
        return loss_dict
        
        # return loss_dict
        