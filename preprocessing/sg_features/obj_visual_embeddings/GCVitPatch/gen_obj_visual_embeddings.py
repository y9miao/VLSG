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

from yaml import scan

workspace_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))))
src_dir = osp.join(workspace_dir, 'src')
sys.path.append(workspace_dir)
sys.path.append(src_dir)

from utils import common, scan3r

def getPatchAnno(gt_anno_2D, patch_w, patch_h, th = 0.5):
    image_h, image_w = gt_anno_2D.shape
    patch_h_size = int(image_h / patch_h)
    patch_w_size = int(image_w / patch_w)
    
    patch_annos = np.zeros((patch_h, patch_w), dtype=np.uint8)
    for patch_h_i in range(patch_h):
        h_start = round(patch_h_i * patch_h_size)
        h_end = round((patch_h_i + 1) * patch_h_size)
        for patch_w_j in range(patch_w):
            w_start = round(patch_w_j * patch_w_size)
            w_end = round((patch_w_j + 1) * patch_w_size)
            patch_size = (w_end - w_start) * (h_end - h_start)
            
            anno = gt_anno_2D[h_start:h_end, w_start:w_end]
            obj_ids, counts = np.unique(anno.reshape(-1), return_counts=True)
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            if(max_count > th*patch_size):
                patch_annos[patch_h_i,patch_w_j] = obj_ids[max_idx]
    return patch_annos

class ObjVisualEmbGen(data.Dataset):
    def __init__(self, cfg, split):
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
            
        # load 2D patch features if use pre-calculated feature
        self.patch_feature_folder = osp.join(self.scans_files_dir, self.cfg.data.img_encoding.feature_dir)
        self.patch_features_path = {}
        for scan_id in self.scan_ids:
            self.patch_features_path[scan_id] = osp.join(self.patch_feature_folder, scan_id + ".pkl")
                
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

    def load3DSceneGraphs(self):
        # load 3D obj semantic annotations
        self.obj_3D_anno = {}
        self.objs_config_file = osp.join(self.scans_files_dir, 'objects.json')
        objs_configs = common.load_json(self.objs_config_file)['scans']
        scans_objs_info = {}
        for scan_item in objs_configs:
            scan_id = scan_item['scan']
            objs_info = scan_item['objects']
            scans_objs_info[scan_id] = objs_info
        for scan_id in self.all_scans_split:
            self.obj_3D_anno[scan_id] = {}
            for obj_item in scans_objs_info[scan_id]:
                obj_id = int(obj_item['id'])
                obj_nyu_category = int(obj_item['nyu40'])
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, obj_nyu_category)
                
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
        # load 2D patch features
        patch_features_path = self.patch_features_path[scan_id]
        patch_features = common.load_pkl_data(patch_features_path)
        
        # iterate over all frames
        ## 2D patch anno
        obj_2D_patch_anno = {}
        if self.img_rotate:
            patch_h = self.image_patch_w
            patch_w = self.image_patch_h
        else:
            patch_h = self.image_patch_h
            patch_w = self.image_patch_w
        for frame_idx in obj_anno_2D:
            obj_2D_anno_f = obj_anno_2D[frame_idx]
            ## process 2D anno
            obj_2D_anno_f = cv2.resize(obj_2D_anno_f, (self.image_resize_w, self.image_resize_h),  # type: ignore
                interpolation=cv2.INTER_NEAREST) # type: ignore
            if self.img_rotate:
                obj_2D_anno_f = obj_2D_anno_f.transpose(1, 0)
                obj_2D_anno_f = np.flip(obj_2D_anno_f, 1)
            obj_2D_patch_anno_f = getPatchAnno(obj_2D_anno_f, patch_w, patch_h, 0.3)
            obj_2D_patch_anno[frame_idx] = obj_2D_patch_anno_f
            obj_ids, counts = np.unique(obj_2D_patch_anno_f.reshape(-1), return_counts=True)
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
                patch_anno_f = obj_2D_patch_anno[frame_idx]
                obj_patch_feature = patch_features[frame_idx][patch_anno_f == obj_id]
                obj_visual_emb[obj_id][frame_idx] = obj_patch_feature
                
        obj_patch_info = {
            'obj_visual_emb': obj_visual_emb,
            'obj_image_votes_topK': obj_image_votes_topK,
        }
        return obj_patch_info
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    from configs import config, update_config
    os.environ['Scan3R_ROOT_DIR'] = "/home/yang/big_ssd/Scan3R/3RScan"
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/preprocessing/sg_features/obj_visual_embeddings/obj_visual_embeddings.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    scan3r_ds = ObjVisualEmbGen(cfg, split='train')
    # scan3r_ds.generateObjVisualEmb()
    obj_patch_info = scan3r_ds.generateObjVisualEmbScan("6a36053b-fa53-2915-9716-6b5361c7791a")
    breakpoint=None