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
        for scan_id in obj_json:
            scan_objs_color = {}
            for obj in obj_json[scan_id]:
                color_str = obj['ply_color'][1:]
                # convert hex color to rgb color
                color_rgb = torch.Tensor([int(color_str[i:i+2], 16) for i in (0, 2, 4)])
                scan_objs_color[obj['id']] = color_rgb
            self.scans_objs_color[scan_id] = scan_objs_color
            
    def visualize(self, data_dict, patch_obj_sim_batch, epoch):
        
        # generate and save visualized image for each data item in batch for each iteration
        batch_size = data_dict['batch_size']
        for batch_i in range(batch_size):
            scan_id = data_dict['scan_ids'][batch_i]
            frame_idx = data_dict['frame_idx'][batch_i]
            
            obj_3D_id2idx = data_dict['obj_3D_id2idx'][batch_i]
            obj_3D_idx2id = {idx: id for id, idx in obj_3D_id2idx.items()}
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i] # (N_P, O)
            
            patch_obj_sim = torch_util.release_cuda(patch_obj_sim_batch[batch_i]) # (N_P, N_O)
            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            matched_obj_confidence = patch_obj_sim.gather(1, matched_obj_idxs)/patch_obj_sim.sum(dim=1) # (N_P)
            
            # dye image patches with color of matched objects
            img = torch_util.release_cuda(data_dict['images'][batch_i]).float() # (H, W, C)
            alpha_map = torch.zeros((img.shape[0], img.shape[1])).float()
            img_color = torch.zeros_like(img).float()
            img_correct = torch.zeros_like(img).float() # show whether the patch is correctly matched
            correct_color = torch.Tensor([0, 255, 0])
            wrong_color = torch.Tensor([255, 0, 0])
            
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
                    
                    img_color[start_h:end_h, start_w:end_h, :] = matched_obj_color
                    img_correct[start_h:end_h, start_w:end_h, :] = correct_color \
                        if e1i_matrix[patch_idx][matched_obj_idxs[patch_idx]] else wrong_color
                    alpha_map[start_h:end_h, start_w:end_h] = matched_obj_confidence[patch_idx]
                    
            # alpha blending
            img_color_blend = img_color*alpha_map.unsqueeze(-1) + img*(1-alpha_map.unsqueeze(-1))
            img_correct_blend = img_correct*alpha_map.unsqueeze(-1) + img*(1-alpha_map.unsqueeze(-1))
            
            # save images
            img_color_blend = img_color_blend.cpu().numpy().astype(np.uint8)
            img_correct_blend = img_correct_blend.cpu().numpy().astype(np.uint8)
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
            
            
            
            
            
            
            