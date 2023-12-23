import os, sys
import os.path as osp
import numpy as np
import cv2
import torch
sys.path.append('..')
sys.path.append('../..')
from utils import common, scan3r, torch_util

class PatchObjectPairVisualizer:
    # not using cuda
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.data.img_encoding.img_rotate:
            self.image_w = self.cfg.data.img_encoding.resize_h
            self.image_h = self.cfg.data.img_encoding.resize_w
            self.image_patch_w = self.cfg.data.img_encoding.patch_h
            self.image_patch_h = self.cfg.data.img_encoding.patch_w
        else:
            self.image_w = self.cfg.data.img.w
            self.image_h = self.cfg.data.img.h
            self.image_patch_w = self.cfg.data.img_encoding.patch_w
            self.image_patch_h = self.cfg.data.img_encoding.patch_h
            
        self.patch_w_size_int = int(self.image_w / self.image_patch_w)
        self.patch_h_size_int = int(self.image_h / self.image_patch_h)
        
        # some traditional dir from SGAligner
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        
        obj_json_filename = 'objects.json'
        obj_json = common.load_json(osp.join(self.data_root_dir, 'files', obj_json_filename))['scans']
        
        # out dir
        self.out_dir = osp.join(cfg.output_dir, "visualize")
        
        # get scan objs color imformation
        self.scans_objs_color = {}
        for objs_info_ps in obj_json:
            scan_id = objs_info_ps['scan']
            scan_objs_color = {}
            for obj_info in objs_info_ps['objects']:
                color_str = obj_info['ply_color'][1:]
                # convert hex color to rgb color
                color_rgb = torch.Tensor([int(color_str[i:i+2], 16) for i in (0, 2, 4)])
                scan_objs_color[int(obj_info['id'])] = color_rgb
            self.scans_objs_color[scan_id] = scan_objs_color
            
    def visualize(self, data_dict, embs, epoch):
        
        # generate and save visualized image for each data item in batch for each iteration
        batch_size = data_dict['batch_size']
        patch_obj_sim_batch = embs['patch_obj_sim']
        for batch_i in range(batch_size):
            scan_id = data_dict['scan_ids'][batch_i]
            frame_idx = data_dict['frame_idxs'][batch_i]
            
            obj_3D_id2idx = data_dict['obj_3D_id2idx'][batch_i]
            obj_3D_idx2id = {idx: id for id, idx in obj_3D_id2idx.items()}
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            
            patch_obj_sim_exp = torch.exp(patch_obj_sim_batch[batch_i]) # (N_P, N_O), no temperature
            matched_obj_idxs = torch.argmax(patch_obj_sim_exp, dim=1).reshape(-1,1) # (N_P)
            matched_obj_confidence = patch_obj_sim_exp.gather(1, matched_obj_idxs).reshape(-1)/patch_obj_sim_exp.sum(dim=1) # (N_P)
            
            # to numpy
            matched_obj_confidence = torch_util.release_cuda(matched_obj_confidence)
            e1i_matrix = torch_util.release_cuda(e1i_matrix) # (N_P, O)
            img = torch_util.release_cuda(data_dict['images'][batch_i]) # (H, W, C)
            
            # record whether patch is labeled 
            patch_labeled = e1i_matrix.sum(axis=1) > 0 # (N_P)
            
            # dye image patches with color of matched objects
            alpha_map = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float)
            img_color = np.zeros_like(img, dtype=np.float)
            img_correct = np.zeros_like(img, dtype=np.float) # show whether the patch is correctly matched
            correct_color = np.array([0, 255, 0])
            wrong_color = np.array([255, 0, 0])
            not_labeled_color = np.array([0, 0, 0])
            
            for patch_h_i in range(self.image_patch_h):
                patch_h_shift = patch_h_i*self.image_patch_w
                for patch_w_j in range(self.image_patch_w):
                    start_h = patch_h_i*self.patch_h_size_int
                    end_h = start_h + self.patch_h_size_int
                    start_w = patch_w_j*self.patch_w_size_int
                    end_w = start_w + self.patch_w_size_int
                    patch_w_shift = patch_w_j
                    patch_idx = patch_h_shift + patch_w_shift
                    matched_obj_idx = matched_obj_idxs[patch_idx].item()
                    matched_obj_id = obj_3D_idx2id[matched_obj_idx]
                    matched_obj_color = self.scans_objs_color[scan_id][matched_obj_id]
                    
                    if not patch_labeled[patch_idx]:
                        img_color[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        img_correct[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        alpha_map[start_h:end_h, start_w:end_w] = 1
                    else:
                        img_color[start_h:end_h, start_w:end_w, :] = matched_obj_color
                        img_correct[start_h:end_h, start_w:end_w, :] = correct_color \
                            if e1i_matrix[patch_idx][matched_obj_idxs[patch_idx]] else wrong_color
                        alpha_map[start_h:end_h, start_w:end_w] = matched_obj_confidence[patch_idx]
                    
            # alpha blending
            alpha_base = 0.5
            alpha_map = np.clip(alpha_map + alpha_base, 0, 1) 
            beta_map = np.clip(1 - alpha_map, 0, 1) 
            img_color_blend = img_color*alpha_map + img*beta_map
            img_correct_blend = img_correct*alpha_map + img*beta_map
            
            # from tensor to numpy
            img_color_blend = torch_util.release_cuda(img_color_blend)
            img_correct_blend = torch_util.release_cuda(img_correct_blend)
            
            # save images
            img_color_blend = img_color_blend.astype(np.uint8)
            img_correct_blend = img_correct_blend.astype(np.uint8)
            # convert rgb to bgr
            img_color_blend = cv2.cvtColor(img_color_blend, cv2.COLOR_RGB2BGR)
            img_correct_blend = cv2.cvtColor(img_correct_blend, cv2.COLOR_RGB2BGR)
            img_color_out_dir = osp.join(self.out_dir, scan_id, frame_idx, 'color')
            img_color_out_path = osp.join(img_color_out_dir, 'epoch_{}.jpg'.format(epoch))
            img_correct_out_dir = osp.join(self.out_dir, scan_id, frame_idx, 'correct')
            img_correct_out_path = osp.join(img_correct_out_dir, 'epoch_{}.jpg'.format(epoch))
            common.ensure_dir(img_color_out_dir)
            common.ensure_dir(img_correct_out_dir)
            cv2.imwrite(img_color_out_path, img_color_blend)
            cv2.imwrite(img_correct_out_path, img_correct_blend)    
            
            
            
            
            
            
            