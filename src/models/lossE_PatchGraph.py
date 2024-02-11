import torch
from torch import nn
import torch.nn.functional as F

def get_loss(cfg):
    loss_type = cfg.train.loss.loss_type
    if loss_type == 'ICLLossBothSidesSumOutLog':
        return ICLLossBothSidesSumOutLog(
            cfg.train.loss.use_temporal,
            cfg.train.loss.temperature, 
            cfg.train.loss.alpha,
            use_depth_aux=cfg.train.auxillary.use_patch_depth,
            depth_aux_weight=cfg.train.auxillary.depth_coeff
            )
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))
    

def get_val_room_retr_loss(cfg):
    return ValidationRoomRetrievalLoss(cfg)


class ICLLossBothSidesSumOutLog(nn.Module):
    def __init__(self, use_temporal, temperature=0.1, alpha = 0.5, epsilon=1e-8, 
                 use_depth_aux = False, depth_aux_weight=0.0):
        super(ICLLossBothSidesSumOutLog, self).__init__()
        self.use_temporal = use_temporal
        self.use_depth_aux = use_depth_aux
        self.temp = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        self.depth_aux_weight = depth_aux_weight
    
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
    
    def forward_item(self, embs, assoc_data_dict, batch_i, key=''):
        # calculate patch loss for each item in the batch
        
        # get patch-object similarity (P_H*P_W, N)
        patch_obj_sim = embs['patch_obj_sim{}'.format(key)][batch_i]
        # get patch-patch similarity (N_P, N_P); N_P = P_H*P_W
        patch_patch_sim = embs['patch_patch_sim{}'.format(key)][batch_i]
        # get object-object similarity (N, N)
        obj_obj_sim = embs['obj_obj_sim{}'.format(key)][batch_i]

        # get matches
        e1i_matrix = assoc_data_dict['e1i_matrix'] # (N_P, N)
        e1j_matrix = assoc_data_dict['e1j_matrix'] # (N_P, N_P)
        e2j_matrix = assoc_data_dict['e2j_matrix'] # (N_P, N)
        # mask out unmatched patches
        e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
        # get 3D matches
        f1i = e1i_matrix.transpose(0,1) # (N, N_P)
        f1j = assoc_data_dict['f1j_matrix'] # (N, N)
        f2j = e2j_matrix.transpose(0,1) # (N, N_P)
        
        if e1i_valid.any():
            loss, matched_obj_labels = self.__class__.calculate_loss(
                self.epsilon, self.temp, self.alpha,
                patch_obj_sim, patch_patch_sim,
                e1i_matrix, e1j_matrix, e2j_matrix, e1i_valid,
                f1i, f1j, f2j, obj_obj_sim)
            loss_batch = loss 
            matched_success_batch = matched_obj_labels
            return loss_batch, matched_success_batch
        else:
            return None, None
        
    def forward_depth_aux(self, embs, data_dict):
        # gt position diff anno
        gt_patch_dist_matrix = data_dict['patch_pos_diff']
        gt_valid_matrix = data_dict['patch_pos_valid_mask']
        
        edges = data_dict['patch_edges']
        # predict position diff
        depth_aux_loss = None
        patch_relative_position = embs['patch_relative_position']
        batch_size = data_dict['batch_size']
        for batch_i in range(data_dict['batch_size']):
            # get predicted position diff
            patch_relative_position_batch = patch_relative_position[batch_i]
            # get gt position diff
            gt_patch_dist_batch = gt_patch_dist_matrix[batch_i][edges[0], edges[1]]
            # get valid mask
            valid_mask_batch = gt_valid_matrix[batch_i][edges[0], edges[1]]
            # calculate loss
            dist_loss_pb = valid_mask_batch.unsqueeze(1) * (gt_patch_dist_batch - patch_relative_position_batch) ** 2
            if depth_aux_loss is None:
                depth_aux_loss = dist_loss_pb.mean()
            else:
                depth_aux_loss += dist_loss_pb.mean()
        depth_aux_loss /= (batch_size + self.epsilon)
        return depth_aux_loss
    
    def forward(self, embs, data_dict):
        # calculate patch loss for each batch
        batch_size = data_dict['batch_size']
        
        loss_batch_NT, loss_batch_T = None, None
        matched_success_batch_NT, matched_success_batch_T = None, None
        
        for batch_i in range(batch_size):
            # dataitem info
            scan_id = data_dict['scan_ids'][batch_i]
            # get assoc data
            assoc_data_dict = data_dict['assoc_data_dict'][batch_i]
            # get loss
            loss_batch, matched_success_batch = self.forward_item(embs, assoc_data_dict, batch_i)
            if loss_batch is not None:
                loss_batch_NT = loss_batch if loss_batch_NT is None else torch.cat([loss_batch_NT, loss_batch])
                matched_success_batch_NT = matched_success_batch if matched_success_batch_NT is None \
                    else torch.cat([matched_success_batch_NT, matched_success_batch])
                    
            # temporal 
            if self.use_temporal:
                assoc_data_dict = data_dict['assoc_data_dict_temp'][batch_i]
                loss_batch, matched_success_batch = self.forward_item(
                    embs, assoc_data_dict, batch_i, key='_temp')
                if loss_batch is not None:
                    loss_batch_T = loss_batch if loss_batch_T is None else torch.cat([loss_batch_T, loss_batch])
                    matched_success_batch_T = matched_success_batch if matched_success_batch_T is None \
                        else torch.cat([matched_success_batch_T, matched_success_batch])
        
        # calculate loss
        ## patch-object loss
        loss_dict = {}
        if not self.use_temporal:
            loss_dict['loss'] = loss_batch_NT.mean() if loss_batch_NT is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
            loss_dict['matched_success_ratio'] = matched_success_batch_NT.mean() if matched_success_batch_NT is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
        else:
            loss_dict['loss'] = (loss_batch_NT.mean() + loss_batch_T.mean()) / 2 if loss_batch_NT is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
            loss_dict['loss_NT'] = loss_batch_NT.mean() if loss_batch_NT is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
            loss_dict['loss_T'] = loss_batch_T.mean() if loss_batch_T is not None \
                else torch.tensor([0.]).to(loss_batch_T.device)
            loss_dict['matched_success_ratio_NT'] = matched_success_batch_NT.mean() if matched_success_batch_NT is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
            loss_dict['matched_success_ratio_T'] = matched_success_batch_T.mean() if matched_success_batch_T is not None \
                else torch.tensor([0.]).to(loss_batch_NT.device)
                
        if self.use_depth_aux:
            depth_aux_loss = self.forward_depth_aux(embs, data_dict)
            loss_dict['depth_aux_loss'] = depth_aux_loss
            loss_dict['loss'] += depth_aux_loss * self.depth_aux_weight
            
        return loss_dict
    
class ValidationRoomRetrievalLoss(nn.Module):
    def __init__(self, cfg):
        super(ValidationRoomRetrievalLoss, self).__init__()
        self.loss = get_loss(cfg)
        self.epsilon_th = cfg.val.room_retrieval.epsilon_th
        self.use_temporal = cfg.train.loss.use_temporal
    
    def forward(self, embs, data_dict):
        # calculate patch loss for each batch
        loss_dict = self.loss(embs, data_dict)
        # room retrieval
        room_retreatl_dict = self.roomRetrieval(embs, data_dict)
        if self.use_temporal:
            room_retreatl_dict_temp = self.roomRetrieval(embs, data_dict, key='_temp')
            room_retreatl_dict.update(room_retreatl_dict_temp)
        
        loss_dict.update(room_retreatl_dict)
        return loss_dict
    
    def roomRetrieval(self, embs, data_dict, key=''):
    
        retrieval_result = {
            'R@1{}'.format(key): 0,
            'R@3{}'.format(key): 0,
            'R@5{}'.format(key): 0,
        }
        
        batch_size = data_dict['batch_size']
        
        total = 0
        for batch_i in range(batch_size):
            # dataitem info
            scan_id = data_dict['scan_ids'][batch_i]
            target_scan_id = data_dict['scan_ids'][batch_i] if key == '' \
                else data_dict['scan_ids_temporal'][batch_i]
            assoc_data_dict = data_dict['assoc_data_dict{}'.format(key)][batch_i]
            # candidate info
            candata_scan_obj_idxs = assoc_data_dict['candata_scan_obj_idxs']
            # get patch-object similarity (P_H*P_W, N)
            patch_obj_sim = embs['patch_obj_sim{}'.format(key)][batch_i]
            
            # get room retrieval result by score
            candidate_room_scores_3 = {} # score = sum( max_j(p_i, o_j)  )
            for candidate_scan_id in candata_scan_obj_idxs.keys():
                candidate_obj_idxs = candata_scan_obj_idxs[candidate_scan_id]
                patch_candidate_obj_sim = patch_obj_sim[:, candidate_obj_idxs]
                matched_candidate_obj_idxs = torch.argmax(patch_candidate_obj_sim, dim=1).reshape(-1,1) # (N_P)
                matched_candidate_obj_sim = patch_candidate_obj_sim.gather(1, matched_candidate_obj_idxs)
                # get scores
                candidate_room_scores_3[candidate_scan_id] = torch.sum(matched_candidate_obj_sim)
                
            # get scan id sorted by given scores
            score_to_used = candidate_room_scores_3
            sorted_scan_id_score = [k for k, v in sorted(score_to_used.items(), key=lambda item: item[1], reverse=True)]
                
            total += 1
            if target_scan_id in sorted_scan_id_score[:1] and score_to_used[target_scan_id] > 0:
                retrieval_result['R@1{}'.format(key)] += 1
            if target_scan_id in sorted_scan_id_score[:3] and score_to_used[target_scan_id] > 0:
                retrieval_result['R@3{}'.format(key)] += 1
            if target_scan_id in sorted_scan_id_score[:5] and score_to_used[target_scan_id] > 0:
                retrieval_result['R@5{}'.format(key)] += 1
    
        retrieval_result['R@1{}'.format(key)]  =  retrieval_result['R@1{}'.format(key)] * 1.0 / total
        retrieval_result['R@3{}'.format(key)]  =  retrieval_result['R@3{}'.format(key)] * 1.0 / total
        retrieval_result['R@5{}'.format(key)]  =  retrieval_result['R@5{}'.format(key)] * 1.0 / total
    
        return retrieval_result
            