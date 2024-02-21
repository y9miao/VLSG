import os
import os.path as osp
import comm
import numpy as np
import random
import albumentations as A
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import cv2
import sys
import scipy
import tqdm
from PIL import Image
# models
import torch
import torch.nn.functional as F
import torch.optim as optim
## clip
import clip
from clip.model import CLIP

from yaml import scan

workspace_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from utils import common, scan3r, scannet_utils

# openmask3d config
top_k = 5
multi_level_expansion_ratio = 0.1
num_of_levels = 3
clip_model_name="ViT-L/14"
dim = 768
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

class ObjVisualEmbGen(data.Dataset):
    def __init__(self, cfg, split, vis=False):
        self.cfg = cfg
        self.undefined = 0
        self.vis = vis

        # scannet scans info
        self.split = split
        scans_info_file = osp.join(cfg.data.root_dir, 'files', 'scans_{}.pkl'.format(split))
        self.rooms_info = common.load_pkl_data(scans_info_file)
        self.scan_ids = []
        self.scan2room = {}
        for room_id in self.rooms_info:
            self.scan_ids += self.rooms_info[room_id]
            for scan_id in self.rooms_info[room_id]:
                self.scan2room[scan_id] = room_id
        
        # get image paths
        self.img_step = cfg.data.img.img_step
        self.data_split_dir = osp.join(cfg.data.root_dir, split)
        self.img_paths = {}
        for scan_id in self.scan_ids:
            img_paths = scannet_utils.load_frame_paths(self.data_split_dir, scan_id, self.img_step)
            self.img_paths[scan_id] = img_paths
                
        # load 2D gt obj id annotation
        self.scans_files_dir = osp.join(cfg.data.root_dir, 'files')
        gt_patch_anno_name = cfg.data.gt_patch
        self.patch_anno_folder = osp.join(self.scans_files_dir, gt_patch_anno_name)
        self.patch_anno = {}
        for scan_id in self.scan_ids:
            patch_anno_scan = common.load_pkl_data(osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id)))
            self.patch_anno[scan_id] = {}
            # filter frames without enough patches
            for frame_idx in self.img_paths[scan_id]:
                if frame_idx in patch_anno_scan:
                    self.patch_anno[scan_id][frame_idx] = patch_anno_scan[frame_idx]
        
        # image info 
        self.image_w = cfg.data.img.w
        self.image_h = cfg.data.img.h
        self.img_resize_w = cfg.data.img.resize_w
        self.img_resize_h = cfg.data.img.resize_h
        self.patch_w = cfg.data.img.patch_w
        self.patch_h = cfg.data.img.patch_h
        self.patch_w_size = self.img_resize_w // self.patch_w
        self.patch_h_size = self.img_resize_h // self.patch_h
            
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
        
        # iterate over all frames
        for frame_idx in self.patch_anno[scan_id]:
            patch_anno_frame = self.patch_anno[scan_id][frame_idx]
            ## process 2D anno
            obj_ids, counts = np.unique(patch_anno_frame, return_counts=True)
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
                    scan_id, frame_idx, obj_id, self.patch_anno[scan_id][frame_idx])    
        obj_patch_info = {
            'obj_visual_emb': obj_visual_emb,
            'obj_image_votes_topK': obj_image_votes_topK,
        }
        return obj_patch_info
    
    def generate_visual_emb(self, scan_id, frame_idx, obj_id, gt_patch_anno):
        # load image
        image_path = self.img_paths[scan_id][frame_idx]
        image = Image.open(image_path)
        image = image.resize((self.img_resize_w, self.img_resize_h))

            
        # get obj mask
        patch_mask = gt_patch_anno == obj_id
        obj_mask =  cv2.resize(patch_mask.astype(np.uint8), (self.img_resize_w, self.img_resize_h),
                               interpolation=cv2.INTER_NEAREST)
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
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/990Pro/scannet_seqs/data"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/sg_features/obj_visual_embeddings/OpenMask3D/openmask3D_emb_scannet.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_ds = ObjVisualEmbGen(cfg, split='val', vis = False)
    scan3r_ds.generateObjVisualEmb()
    # obj_patch_info = scan3r_ds.generateObjVisualEmbScan("6a36053b-fa53-2915-9716-6b5361c7791a")
    breakpoint=None