import torch
from torch import nn
import torch.nn.functional as F

def get_loss(cfg):
    loss_type = cfg.train.loss.loss_type
    if loss_type == 'ICLLoss':
        return ICLLoss(cfg.train.loss.temperature, cfg.train.loss.alpha)
    elif loss_type == 'ICLLossBothSides':
        return ICLLossBothSides(cfg.train.loss.temperature, cfg.train.loss.alpha)
    elif loss_type == 'ICLLossBothSidesSumOutLog':
        return ICLLossBothSidesSumOutLog(cfg.train.loss.temperature, cfg.train.loss.alpha)
    elif loss_type == 'TripletLoss':
        return TripletLoss(cfg.train.loss.margin)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))
    
def get_val_loss(cfg):
    return ValidationLoss(cfg.train.loss.margin, cfg.train.loss.temperature, 
                          cfg.train.loss.alpha, cfg.train.loss.epsilon, cfg.train.loss.loss_type)

class ICLLoss(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(ICLLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        
    @staticmethod
    def calculate_loss(epsilon, temp, 
                       patch_obj_sim, patch_patch_sim,
                       e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid):
        # calculate exp similarity of patch-object and patch-patch
        patch_obj_sim_exp = torch.exp(patch_obj_sim / temp)
        patch_patch_sim_exp = torch.exp(patch_patch_sim / temp)
        delta_E1i_E2i = (e1i_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        
        # calculate loss
        sum_delta_E1i_E1j = (e1j_matrix * patch_patch_sim_exp).sum(dim=-1) # (N_P)
        sum_delta_E1i_E2j = (e2j_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        loss_patch_side = delta_E1i_E2i / (delta_E1i_E2i + sum_delta_E1i_E1j + sum_delta_E1i_E2j + epsilon)
        
        loss, matched_obj_labels = None, None
        if e1i_valid.any():
            loss_patch_side = loss_patch_side[e1i_valid]
            loss = -torch.log(loss_patch_side + epsilon)
            
            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs) # (N_P, 1)
            matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
            matched_obj_labels = matched_obj_labels[e1i_valid]
        return loss, matched_obj_labels
            
    def forward(self, embs, data_dict):
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]

        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # get patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            
            # get matches
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            
            if e1i_valid.any():
                loss, matched_obj_labels = self.__class__.calculate_loss(
                    self.epsilon, self.temp,patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid)
                loss_batch = loss if loss_batch is None else torch.cat((loss_batch, loss), dim=0)
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
                    
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        
        return loss_dict
    
class ICLLossBothSides(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(ICLLossBothSides, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        
    @staticmethod
    def calculate_loss(epsilon, temp, alpha, 
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim):
        # calculate exp similarity of patch-object and patch-patch
        patch_obj_sim_exp = torch.exp(patch_obj_sim / temp)
        patch_patch_sim_exp = torch.exp(patch_patch_sim / temp)
        
        # calculate loss
        delta_E1i_E2i = (e1i_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        sum_delta_E1i_E1j = (e1j_matrix * patch_patch_sim_exp).sum(dim=-1) # (N_P)
        sum_delta_E1i_E2j = (e2j_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        loss_patch_side = delta_E1i_E2i / (delta_E1i_E2i + sum_delta_E1i_E1j 
                                            + sum_delta_E1i_E2j + epsilon)
    
        # calculate 3D-to-2D loss
        obj_obj_sim_exp = torch.exp(obj_obj_sim / temp)
        delta_F1i_F2i = delta_E1i_E2i # (N_P)
        gt_matched_obj_idxs = torch.argmax(e1i_matrix, dim=1).reshape(-1,1) # (N_P,1)
        delta_F1i_F1j_all_objs = (f1j * obj_obj_sim_exp).sum(dim=-1).reshape(-1,1) # (O,1)
        sum_delta_F1i_F1j = delta_F1i_F1j_all_objs.gather(0, gt_matched_obj_idxs).reshape(-1) # (N_P,1)
        obj_patch_sim_exp = patch_obj_sim_exp.transpose(0,1) # (O, N_P)
        delta_F1i_F2j_all_objs = (f2j * obj_patch_sim_exp).sum(dim=-1).reshape(-1,1) # (O,1)
        sum_delta_F1i_F2j = delta_F1i_F2j_all_objs.gather(0, gt_matched_obj_idxs).reshape(-1) # (N_P,1)
        loss_obj_side = delta_F1i_F2i / (delta_F1i_F2i + sum_delta_F1i_F1j + sum_delta_F1i_F2j + epsilon) # (N_P)
        
        loss, matched_obj_labels = None, None
        if e1i_valid.any():
            loss_both_sides = loss_patch_side[e1i_valid] * alpha + \
                loss_obj_side[e1i_valid] * (1 - alpha)
            loss = -torch.log(loss_both_sides + epsilon)
            
            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs) # (N_P, 1)
            matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
            matched_obj_labels = matched_obj_labels[e1i_valid]
        return loss, matched_obj_labels
    
    def forward(self, embs, data_dict):
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]
        obj_obj_sim_list = embs['obj_obj_sim'] # B - [O, O]
        
        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # get patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            
            # get matches
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            # mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            # get 3D matches
            obj_obj_sim = obj_obj_sim_list[batch_i] # (O, O)
            num_objs = e1i_matrix.shape[1]
            f1i = e1i_matrix.transpose(0,1) # (O, N_P)
            # f1j = (torch.ones((num_objs, num_objs),device = obj_obj_sim.device) - \
            #     torch.eye(num_objs, device = obj_obj_sim.device)) # (O, O)
            f1j = data_dict['f1j_matrix_list'][batch_i] # (O, O)
            f2j = e2j_matrix.transpose(0,1) # (O, N_P)
            
            if e1i_valid.any():
                loss, matched_obj_labels = self.__class__.calculate_loss(
                    self.epsilon, self.temp, self.alpha,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                
                loss_batch = loss if loss_batch is None else torch.cat((loss_batch, loss), dim=0)
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        return loss_dict
    
    
class ICLLossBothSidesSumOutLog(nn.Module):
    def __init__(self, temperature=0.1, alpha = 0.5, epsilon=1e-8):
        super(ICLLossBothSidesSumOutLog, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
    
    @staticmethod
    def calculate_loss(epsilon, temp, alpha, 
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim):
        # calculate exp similarity of patch-object and patch-patch
        patch_obj_sim_exp = torch.exp(patch_obj_sim / temp)
        patch_patch_sim_exp = torch.exp(patch_patch_sim / temp)
        
        # calculate loss
        delta_E1i_E2i = (e1i_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        sum_delta_E1i_E1j = (e1j_matrix * patch_patch_sim_exp).sum(dim=-1) # (N_P)
        sum_delta_E1i_E2j = (e2j_matrix * patch_obj_sim_exp).sum(dim=-1) # (N_P)
        loss_patch_side = delta_E1i_E2i / (delta_E1i_E2i + sum_delta_E1i_E1j 
                                            + sum_delta_E1i_E2j + epsilon)
    
        # calculate 3D-to-2D loss
        obj_obj_sim_exp = torch.exp(obj_obj_sim / temp)
        delta_F1i_F2i = delta_E1i_E2i # (N_P)
        gt_matched_obj_idxs = torch.argmax(e1i_matrix, dim=1).reshape(-1,1) # (N_P,1)
        delta_F1i_F1j_all_objs = (f1j * obj_obj_sim_exp).sum(dim=-1).reshape(-1,1) # (O,1)
        sum_delta_F1i_F1j = delta_F1i_F1j_all_objs.gather(0, gt_matched_obj_idxs).reshape(-1) # (N_P,1)
        obj_patch_sim_exp = patch_obj_sim_exp.transpose(0,1) # (O, N_P)
        delta_F1i_F2j_all_objs = (f2j * obj_patch_sim_exp).sum(dim=-1).reshape(-1,1) # (O,1)
        sum_delta_F1i_F2j = delta_F1i_F2j_all_objs.gather(0, gt_matched_obj_idxs).reshape(-1) # (N_P,1)
        loss_obj_side = delta_F1i_F2i / (delta_F1i_F2i + sum_delta_F1i_F1j + sum_delta_F1i_F2j + epsilon) # (N_P)
        
        loss, matched_obj_labels = None, None
        if e1i_valid.any():
            loss = -torch.log(loss_patch_side[e1i_valid] + epsilon) * alpha + \
                -torch.log(loss_obj_side[e1i_valid] + epsilon) * (1 - alpha)

            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs) # (N_P, 1)
            matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
            matched_obj_labels = matched_obj_labels[e1i_valid]
        return loss, matched_obj_labels
    
    def forward(self, embs, data_dict):
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]
        obj_obj_sim_list = embs['obj_obj_sim'] # B - [O, O]
        
        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # get patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            
            # get matches
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            # mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            # get 3D matches
            obj_obj_sim = obj_obj_sim_list[batch_i] # (O, O)
            num_objs = e1i_matrix.shape[1]
            f1i = e1i_matrix.transpose(0,1) # (O, N_P)
            # f1j = (torch.ones((num_objs, num_objs),device = obj_obj_sim.device) - \
            #     torch.eye(num_objs, device = obj_obj_sim.device)) # (O, O)
            f1j = data_dict['f1j_matrix_list'][batch_i] # (O, O)
            f2j = e2j_matrix.transpose(0,1) # (O, N_P)
            
            if e1i_valid.any():
                loss, matched_obj_labels = self.__class__.calculate_loss(
                    self.epsilon, self.temp, self.alpha,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                
                loss_batch = loss if loss_batch is None else torch.cat((loss_batch, loss), dim=0)
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        return loss_dict
    
class TripletLoss(nn.Module):
    def __init__(self, margin = 0.1, epsilon=1e-8):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon
        
    @staticmethod
    def calculate_loss(epsilon, margin,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim):
        
        # calculate triplet loss
        loss, matched_obj_labels = None, None
        if e1i_valid.any():
            loss_this_batch = patch_obj_sim*e2j_matrix - \
                patch_obj_sim*e1i_matrix.sum(dim=-1).reshape(-1,1) 
            loss_this_batch = loss_this_batch[e1i_valid]
            loss_this_batch = loss_this_batch[e2j_matrix[e1i_valid].nonzero(as_tuple=True)] # cos_sim of negative - cos_sim of positive
            loss_this_batch = loss_this_batch + margin
            loss_this_batch = loss_this_batch[loss_this_batch > 0]

            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs) # (N_P, 1)
            matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
            matched_obj_labels = matched_obj_labels[e1i_valid]
        return loss_this_batch, matched_obj_labels
    
    def forward(self, embs, data_dict):
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]
        obj_obj_sim_list = embs['obj_obj_sim'] # B - [O, O]
        
        # calculate patch loss for each batch
        loss_batch = None
        matched_success_batch = None
        
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # get patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            
            # get matches
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            # mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            # get 3D matches
            obj_obj_sim = obj_obj_sim_list[batch_i] # (O, O)
            num_objs = e1i_matrix.shape[1]
            f1i = e1i_matrix.transpose(0,1) # (O, N_P)
            # f1j = (torch.ones((num_objs, num_objs),device = obj_obj_sim.device) - \
            #     torch.eye(num_objs, device = obj_obj_sim.device)) # (O, O)
            f1j = data_dict['f1j_matrix_list'][batch_i] # (O, O)
            f2j = e2j_matrix.transpose(0,1) # (O, N_P)
            
            if e1i_valid.any():
                loss, matched_obj_labels = self.__class__.calculate_loss(
                    self.epsilon, self.margin,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                
                loss_batch = loss if loss_batch is None else torch.cat((loss_batch, loss), dim=0)
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.
        }
        return loss_dict
    
class ValidationLoss(nn.Module):
    def __init__(self, margin = 0.1, temperature=0.1, alpha = 0.5, epsilon=1e-8, loss_type='ICLLoss'):
        super(ValidationLoss, self).__init__()
        self.margin = margin
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        self.loss_type = loss_type
    
    def forward(self, embs, data_dict):
        patch_obj_sim_list = embs['patch_obj_sim'] # B - [P_H*P_W, O]
        patch_patch_sim_list = embs['patch_patch_sim'] # B - [P_H*P_W, P_H*P_W]
        obj_obj_sim_list = embs['obj_obj_sim'] # B - [O, O]
        
        # calculate patch loss for each batch
        loss_batch = None
        loss_batch_one_side = None
        loss_batch_both_sides = None
        loss_batch_both_sides2 = None
        loss_batch_triplet = None
        matched_success_batch = None
        
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            # get patch-object similarity (P_H*P_W, O)
            patch_obj_sim = patch_obj_sim_list[batch_i]
            # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
            patch_patch_sim = patch_patch_sim_list[batch_i]
            
            # get matches
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            e1j_matrix = data_dict['e1j_matrix_list'][batch_i] # (N_P, N_P)
            e2j_matrix = data_dict['e2j_matrix_list'][batch_i] # (N_P, O)
            # mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            # get 3D matches
            obj_obj_sim = obj_obj_sim_list[batch_i] # (O, O)
            num_objs = e1i_matrix.shape[1]
            f1i = e1i_matrix.transpose(0,1) # (O, N_P)
            f1j = (torch.ones((num_objs, num_objs),device = obj_obj_sim.device) - \
                torch.eye(num_objs, device = obj_obj_sim.device)) # (O, O)
            f2j = e2j_matrix.transpose(0,1) # (O, N_P)
            
            if e1i_valid.any():
                loss_one_side, matched_obj_labels = ICLLoss.calculate_loss(
                    self.epsilon, self.temp,patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid)
                loss_both_sides, matched_obj_labels = ICLLossBothSides.calculate_loss(
                    self.epsilon, self.temp, self.alpha,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                loss_both_sides2, matched_obj_labels = ICLLossBothSidesSumOutLog.calculate_loss(
                    self.epsilon, self.temp, self.alpha,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                loss_triplet, matched_obj_labels = TripletLoss.calculate_loss(
                    self.epsilon, self.margin,
                    patch_obj_sim, patch_patch_sim,
                    e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                    f1i, f1j, f2j, obj_obj_sim)
                
                loss_batch_one_side = loss_one_side if loss_batch_one_side is None else torch.cat((loss_batch_one_side, loss_one_side), dim=0)
                loss_batch_both_sides = loss_both_sides if loss_batch_both_sides is None else torch.cat((loss_batch_both_sides, loss_both_sides), dim=0)
                loss_batch_both_sides2 = loss_both_sides2 if loss_batch_both_sides2 is None else torch.cat((loss_batch_both_sides2, loss_both_sides2), dim=0)
                loss_batch_triplet = loss_triplet if loss_batch_triplet is None else torch.cat((loss_batch_triplet, loss_triplet), dim=0)
                matched_success_batch = matched_obj_labels if matched_success_batch is None \
                    else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
        
        if self.loss_type == 'ICLLoss':
            loss_batch = loss_batch_one_side
        elif self.loss_type == 'ICLLossBothSides':
            loss_batch = loss_batch_both_sides
        elif self.loss_type == 'ICLLossBothSidesSumOutLog':
            loss_batch = loss_batch_both_sides2
        elif self.loss_type == 'TripletLoss':
            loss_batch = loss_batch_triplet
        else:
            raise ValueError('Unknown loss type: {}'.format(self.loss_type))
        
        loss_dict =  {
            'loss': loss_batch.mean() if loss_batch is not None else 0.,
            'matched_success_ratio': matched_success_batch.float().mean() \
                if loss_batch is not None else 0.,
            'ICLLoss': loss_batch_one_side.mean() if loss_batch_one_side is not None else 0.,
            'ICLLossBothSides': loss_batch_both_sides.mean() if loss_batch_both_sides is not None else 0.,
            'ICLLossBothSidesSumOutLog': loss_batch_both_sides2.mean() if loss_batch_both_sides2 is not None else 0.,
            'TripletLoss': loss_batch_triplet.mean() if loss_batch_triplet is not None else 0.,
        }
        
        return loss_dict
