import os
import os.path as osp
import comm
import numpy as np
import random
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image
import cv2
import sys
import scipy
import tqdm
# models
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torchvision import transforms as T
from torchvision import transforms as tvf
from yaml import scan
from typing import Literal, Tuple, List, Union
workspace_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)
from utils import common, scan3r

# Dino v2
from dinov2_utils import DinoV2ExtractFeatures
from dataclasses import dataclass
@dataclass
class LocalArgs:
    """
        Local arguments for the program
    """
    # Dino_v2 properties (parameters)
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_c: int = 32
    # device
    device = torch.device("cuda")
larg = tyro.cli(LocalArgs)

# openmask3d multi-level functions
def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1

def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2 , y2  = mask2box(mask)
    if level == 0:
        return x1, y1, x2 , y2
    shape = mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

# multiview config
top_k = 5
multi_level_expansion_ratio = 0.2
num_of_levels = 3
feat_dim = 1536

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
        self.topk = cfg.data.obj_img.topk
            
        # out obj visual emb dir
        self.obj_visual_emb_dir = osp.join(self.scans_files_dir, self.cfg.data.obj_img.name)
        common.ensure_dir(self.obj_visual_emb_dir)
        
        # get device 
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        # load DinoV2 model
        self.larg = larg
        desc_layer = larg.desc_layer
        desc_facet = larg.desc_facet
        device = larg.device
        self.device = larg.device
        # Dinov2 extractor
        if "extractor" in globals():
            print(f"Extractor already defined, skipping")
        else:
            self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
                desc_facet, use_cls = True , device=device)

        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
                
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
        if self.vis:
            image.show()
            cv2.imshow("obj_2D_anno_f_rot", np.array(obj_2D_anno_f_rot))
        # get obj mask
        obj_mask = obj_2D_anno_f_rot == obj_id
        # extract multi-level crop dinov2 features
        images_crops = []
        for level in range(num_of_levels):
            mask_tensor = torch.from_numpy(obj_mask).to(self.device).float()
            x1, y1, x2, y2 = mask2box_multi_level(mask_tensor, level, multi_level_expansion_ratio)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((224, 224), Image.BICUBIC)
            img_pt = self.base_tf(cropped_img).to(self.device)
            images_crops.append(img_pt)
        # extract clip emb
        if(len(images_crops) > 0):
            image_input =torch.stack(images_crops)
            with torch.no_grad():
                ret = self.extractor(image_input) # [num_levels, 1+num_patches, desc_dim]  
                # get cls token
                cls_token = ret[:, 0, :]
                # get mean of all patches
                mean_patch = cls_token.mean(dim=0)
        return mean_patch.cpu().detach().numpy()
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_ds = ObjVisualEmbGen(cfg, split='train', vis = False)
    scan3r_ds.generateObjVisualEmb()
    # obj_patch_info = scan3r_ds.generateObjVisualEmbScan("6a36053b-fa53-2915-9716-6b5361c7791a")
    breakpoint=None