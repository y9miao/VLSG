import os
import os.path as osp
import comm
import numpy as np
import random
import torch
import torch.utils.data as data
import cv2
import sys
import scipy
import tqdm
from PIL import Image
# models
import torch
import torch.nn.functional as F
import torch.optim as optim
from yaml import scan

# vlsg utils
workspace_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
vlsg_utils_dir = osp.join(workspace_dir, 'utils')
sys.path.append(vlsg_utils_dir)
import common, scan3r

# uni3d settings
uni3d_wk_dir = "/home/yang/big_ssd/Scan3R/ThirdParty/Uni3D"
sys.path.insert(0, uni3d_wk_dir)
import open_clip
import models.uni3d as models

class ObjVisualEmbGen(data.Dataset):
    def __init__(self, cfg, split, vis=False):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
        # set random seed
        self.seed = cfg.seed
        random.seed(self.seed)
        
        # sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)

        # 2D images
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.vis = vis
        
        # 2D_patches
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.step = self.cfg.data.img.img_step

        # scene_img_dir
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        # if split is val, then use all object from other scenes as negative samples
        # if room_retrieval, then use load additional data items
        
        # scans info
        self.temporal = cfg.data.temporal
        self.rescan = cfg.data.rescan
        scan_info_file = osp.join(self.scans_files_dir, '3RScan.json')
        all_scan_data = common.load_json(scan_info_file)
        self.refscans2scans = {}
        self.scans2refscans = {}
        self.all_scans_split = []
        for scan_data in all_scan_data:
            ref_scan_id = scan_data['reference']
            self.refscans2scans[ref_scan_id] = [ref_scan_id]
            self.scans2refscans[ref_scan_id] = ref_scan_id
            for scan in scan_data['scans']:
                self.refscans2scans[ref_scan_id].append(scan['reference'])
                self.scans2refscans[scan['reference']] = ref_scan_id
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        self.all_scans_split = []
        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            self.all_scans_split += self.refscans2scans[ref_scan]
        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split
            
        ## image patches 
        self.image_path = {}
        for scan_id in self.scan_ids:
            self.image_path[scan_id] = scan3r.load_frame_paths(self.data_root_dir, scan_id)
                
        # load 2D gt obj id annotation
        self.gt_2D_anno_folder = osp.join(self.scans_files_dir, 'gt_projection/obj_id_pkl')
        self.obj_2D_annos_pathes = {}
        for scan_id in self.scan_ids:
            anno_2D_file = osp.join(self.gt_2D_anno_folder, "{}.pkl".format(scan_id))
            self.obj_2D_annos_pathes[scan_id] = anno_2D_file
            
        # obj visual emb config
        self.topk = top_k
            
        # out obj visual emb dir
        self.obj_visual_emb_dir = osp.join(self.scans_files_dir, self.cfg.data.obj_img.name)
        common.ensure_dir(self.obj_visual_emb_dir)
        
        # get device 
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        # load clip model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, jit=False)
        self.clip_model.to(torch.device('cuda'))
        self.clip_model.eval()
                
    def generateObjVisualEmb(self):
        for scan_id in tqdm.tqdm(self.scan_ids):
            obj_patch_info = self.generateObjVisualEmbScan(scan_id)
            obj_visual_emb_file = osp.join(self.obj_visual_emb_dir, "{}.pkl".format(scan_id))
            common.write_pkl_data(obj_patch_info, obj_visual_emb_file)
            
    def generateObjVisualEmbScan(self, scan_id):
        obj_image_votes = {}
        
        # load gt 2D obj anno
        obj_anno_2D_file = self.obj_2D_annos_pathes[scan_id]
        obj_anno_2D = common.load_pkl_data(obj_anno_2D_file)
        
        # iterate over all frames
        for frame_idx in obj_anno_2D:
            obj_2D_anno_frame = obj_anno_2D[frame_idx]
            ## process 2D anno
            obj_ids, counts = np.unique(obj_2D_anno_frame, return_counts=True)
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                count = counts[idx]
                if obj_id == self.undefined:
                    continue
                if obj_id not in obj_image_votes:
                    obj_image_votes[obj_id] = {}
                if frame_idx not in obj_image_votes[obj_id]:
                    obj_image_votes[obj_id][frame_idx] = 0
                obj_image_votes[obj_id][frame_idx] = count
        ## select top K frames for each obj
        obj_image_votes_topK = {}
        for obj_id in obj_image_votes:
            obj_image_votes_topK[obj_id] = []
            obj_image_votes_f = obj_image_votes[obj_id]
            sorted_frame_idxs = sorted(obj_image_votes_f, key=obj_image_votes_f.get, reverse=True)
            if len(sorted_frame_idxs) > self.cfg.data.obj_img.topk:
                obj_image_votes_topK[obj_id] = sorted_frame_idxs[:self.cfg.data.obj_img.topk]
            else:
                obj_image_votes_topK[obj_id] = sorted_frame_idxs
        ## get obj visual emb
        obj_visual_emb = {}
        for obj_id in obj_image_votes_topK:
            obj_image_votes_topK_frames = obj_image_votes_topK[obj_id]
            obj_visual_emb[obj_id] = {}
            for frame_idx in obj_image_votes_topK_frames:
                obj_visual_emb[obj_id][frame_idx] = self.generate_visual_emb(
                    scan_id, frame_idx, obj_id, obj_anno_2D[frame_idx])    
        obj_patch_info = {
            'obj_visual_emb': obj_visual_emb,
            'obj_image_votes_topK': obj_image_votes_topK,
        }
        return obj_patch_info
    
    def generate_visual_emb(self, scan_id, frame_idx, obj_id, gt_anno):
        if self.img_rotate:
            obj_2D_anno_f_rot = gt_anno.transpose(1, 0)
            obj_2D_anno_f_rot = np.flip(obj_2D_anno_f_rot, 1)
        # load image
        image_path = self.image_path[scan_id][frame_idx]
        image = Image.open(image_path)
        if self.img_rotate:
            image = image.transpose(Image.ROTATE_270)
        
            
        # get obj mask
        obj_mask = obj_2D_anno_f_rot == obj_id
        # extract multi-level crop clip embs
        images_crops = []
        for level in range(num_of_levels):
            mask_tensor = torch.from_numpy(obj_mask).to(self.device).float()
            x1, y1, x2, y2 = mask2box_multi_level(mask_tensor, level, multi_level_expansion_ratio)
            cropped_img = image.crop((x1, y1, x2, y2))
            if self.vis:
                cropped_img.show()
            cropped_img_processed = self.clip_preprocess(cropped_img).to(self.device)
            images_crops.append(cropped_img_processed)
        # extract clip emb
        if(len(images_crops) > 0):
            image_input =torch.stack(images_crops)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input.float())
                image_features /= image_features.norm(dim=-1, keepdim=True) #normalize
                image_features = image_features.mean(axis=0)
        return image_features.cpu().detach().numpy()

        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/990Pro/scannet_seqs/datan"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/sg_features/obj_visual_embeddings/OpenMask3D/openmask3D_emb_scannet.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_ds = ObjVisualEmbGen(cfg, split='val', vis = True)
    scan3r_ds.generateObjVisualEmb()
    # obj_patch_info = scan3r_ds.generateObjVisualEmbScan("6a36053b-fa53-2915-9716-6b5361c7791a")
    breakpoint=None