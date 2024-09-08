import os
import os.path as osp
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from yaml import scan

src_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
ws_dir = osp.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)

from utils import common, scan3r, scannet_utils

class ScannetPatchObjDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        
        # undefined patch anno id
        self.undefined = 0
        
        # set random seed
        self.seed = cfg.seed
        random.seed(self.seed)
        
        # scannet scans info
        self.split = split
        self.sgaligner_modules = cfg.sgaligner.modules
        scans_info_file = osp.join(cfg.data.root_dir, 'files', 'scans_{}.pkl'.format(split))
        self.rooms_info = common.load_pkl_data(scans_info_file)
        self.scan_ids = []
        self.scan2room = {}
        for room_id in self.rooms_info:
            self.scan_ids += self.rooms_info[room_id]
            for scan_id in self.rooms_info[room_id]:
                self.scan2room[scan_id] = room_id
        self.room2scans = self.rooms_info
        
        # cross scenes cfg
        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes
        self.num_negative_samples = cfg.data.cross_scene.num_negative_samples
        self.use_tf_idf = cfg.data.cross_scene.use_tf_idf
        
        # room retrieval cfg
        self.retrieval = cfg.val.room_retrieval.retrieval
                
        # 2D img cfgs
        self.img_step = cfg.data.inference_step
        self.patch_w = self.cfg.data.img_encoding.patch_w
        self.patch_h = self.cfg.data.img_encoding.patch_h
        self.num_patch = self.patch_h * self.patch_w
        # 2D features patch
        self.root_dir = cfg.data.root_dir
        self.split_folder = osp.join(self.root_dir, self.split)
        feature_folder = osp.join(self.root_dir, 'files', cfg.data.feature_2D_name)
        self.frame_idxs = {}
        self.features_path = {}
        for scan_id in self.scan_ids:
            frame_idxs = scannet_utils.load_frame_idxs(self.split_folder, scan_id, self.img_step)
            self.features_path[scan_id] = {}
            for frame_idx in frame_idxs:
                feature_file = osp.join(feature_folder, scan_id, "{}.npy".format(frame_idx))
                self.features_path[scan_id][frame_idx] = feature_file
                
                self.frame_idxs[scan_id] = frame_idxs
        self.img_patch_feat_dim = self.cfg.sgaligner.model.img_patch_feat_dim
        
        # load patch anno
        self.scans_files_dir = osp.join(self.root_dir, 'files')
        self.patch_anno = {}
        patch_anno_name = cfg.data.gt_patch
        patch_anno_th = cfg.data.gt_patch_th
        self.patch_anno_folder = osp.join(self.scans_files_dir, patch_anno_name)
        for scan_id in self.scan_ids:
            patch_anno_scan = common.load_pkl_data(osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id)))
            self.patch_anno[scan_id] = {}
            # filter frames without enough patches
            for frame_idx in self.frame_idxs[scan_id]:
                if frame_idx in patch_anno_scan:
                    num_valid_patches = np.sum(patch_anno_scan[frame_idx]!=0)
                    if num_valid_patches * 1.0 / self.num_patch > patch_anno_th:
                        self.patch_anno[scan_id][frame_idx] = patch_anno_scan[frame_idx]

        # 3D scene graph info
        ## load 3D scene graph information
        self.load3DSceneGraphs()
        ## load obj_visual features
        self.img_patch_feat_dim = self.cfg.sgaligner.model.img_patch_feat_dim
        obj_img_patch_name = self.cfg.data.scene_graph.obj_img_patch
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.obj_img_patches_scan_tops = {}
        if 'img_patch' in self.sgaligner_modules:
            for scan_id in self.scan_ids:
                obj_visual_file = osp.join(self.scans_files_dir, obj_img_patch_name, scan_id+'.pkl')
                self.obj_img_patches_scan_tops[scan_id] = common.load_pkl_data(obj_visual_file)
                
        ## 3D pc augment
        self.use_aug = cfg.train.data_aug.use_aug
        self.elastic_distortion = scannet_utils.ElasticDistortion(
            apply_distorsion=cfg.train.data_aug.use_aug_3D,
            granularity=cfg.train.data_aug.pcs.granularity,
            magnitude=cfg.train.data_aug.pcs.magnitude)
        
        # fix candidate scan for val&test split for room retrieval
        if self.split == 'val' or self.split == 'test':
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = scannet_utils.sampleCandidateScenesForEachScan(
                    scan_id, self.scan_ids, self.rooms_info, self.scan2room, self.num_scenes)
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()

    def load3DSceneGraphs(self):
        # load scene graph
        self.use_pred_sg = self.cfg.data.scene_graph.use_predicted
        self.pc_resolution = self.cfg.sgaligner.pc_res
        sg_folder_name = 'scene_graph_fusion' if self.use_pred_sg else 'gt_scan3r'
        sg_modules = self.cfg.sgaligner.modules
        self.sg_modules = sg_modules
        ## scene graph
        self.scene_graphs = {}
        ## 3D obj info
        self.obj_3D_anno = {}
        for scan_id in self.scan_ids:
            sg_folder_scan = osp.join(self.split_folder, scan_id, sg_folder_name)
            # Centering
            points = scan3r.load_plydata_npy(osp.join(sg_folder_scan, 'data.npy'))
            pcl_center = np.mean(points, axis=0)
            # scene graph info
            scene_graph_dict = common.load_pkl_data(osp.join(sg_folder_scan, '{}.pkl'.format(scan_id)))
            object_ids = scene_graph_dict['objects_id']
            global_object_ids = scene_graph_dict['objects_cat']
            object_points = scene_graph_dict['obj_points'][self.pc_resolution] - pcl_center
            # load data to tensor
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
            
            # aggreate data 
            data_dict = {} 
            if 'rel' in sg_modules or 'gat' in sg_modules:
                edges = scene_graph_dict['edges']
                edges = torch.from_numpy(edges)
                if 'bow_vec_object_edge_feats' in scene_graph_dict:
                    bow_vec_obj_edge_feats = torch.from_numpy(scene_graph_dict['bow_vec_object_edge_feats'])
                else:
                    rel_dim = self.cfg.sgaligner.model.rel_dim
                    bow_vec_obj_edge_feats = torch.zeros(edges.shape[0], rel_dim)
                data_dict['graph_per_edge_count'] = np.array([edges.shape[0]])
                data_dict['tot_bow_vec_object_edge_feats'] = bow_vec_obj_edge_feats
                rel_pose = torch.from_numpy(scene_graph_dict['rel_trans'])
                data_dict['tot_rel_pose'] = rel_pose
                data_dict['edges'] = edges    
            if 'attr' in sg_modules:
                if 'bow_vec_object_attr_feats' in scene_graph_dict:
                    bow_vec_obj_attr_feats = torch.from_numpy(scene_graph_dict['bow_vec_object_attr_feats'])
                else:
                    attri_dim = self.cfg.sgaligner.model.attr_dim
                    bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], attri_dim)
                data_dict['tot_bow_vec_object_attr_feats'] = bow_vec_obj_attr_feats

            data_dict['obj_ids'] = object_ids
            data_dict['tot_obj_pts'] = object_points
            data_dict['graph_per_obj_count'] = np.array([object_points.shape[0]])
            data_dict['tot_obj_count'] = object_points.shape[0]
            data_dict['scene_ids'] = [scan_id]        
            data_dict['pcl_center'] = pcl_center
            # get scene graph
            self.scene_graphs[scan_id] = data_dict
            
            # obj info 
            self.obj_3D_anno[scan_id] = {}
            for idx, obj_id in enumerate(object_ids):
                self.obj_3D_anno[scan_id][obj_id] = (scan_id, obj_id, global_object_ids[idx])
            breakpoint = 1

    def sampleCandidateScenesForScans(self, scan_ids, num_scenes):
        candidate_scans = {}
        # ref scans of input scans
        room_ids = [self.scan2room[scan_id] for scan_id in scan_ids]
        room_ids = list(set(room_ids))
        num_rooms= len(room_ids)
        num_scans_to_be_sampled = num_scenes
        additional_candidate_sample_pool = [scan for scan in self.scan_ids if self.scan2room[scan] not in room_ids]
        additional_candidates = random.sample(additional_candidate_sample_pool, num_scans_to_be_sampled)
        for scan_id in scan_ids:
            candidate_scans[scan_id] = list(set(additional_candidates))      
        candidate_scans_all = list(set([scan for scan_list in candidate_scans.values() for scan in scan_list]))
        union_scans = list(set(scan_ids + candidate_scans_all))
        return candidate_scans, union_scans
    def sampleCrossRooms(self, scan_id):
        candidate_scans = []
        room_id = self.scan2room[scan_id]
        for scan in self.rooms_info[room_id]:
            if scan != scan_id:
                candidate_scans.append(scan)
        if len(candidate_scans) == 0:
            return None
        else:
            sampled_scan = random.sample(candidate_scans, 1)[0]
            return sampled_scan
            
    def generateDataItems(self):
        data_items = []
        # iterate over scans
        for scan_id in self.scan_ids:
            # skip scans without rescans for val&test split
            if self.split != 'train':
                rescan_id = self.sampleCrossRooms(scan_id)
                if rescan_id is None:
                    continue
            if self.split != 'train' and self.retrieval:
                target_scan_id = rescan_id
            else:
                target_scan_id = scan_id
            
            # iterate over frames
            for frame_idx in self.frame_idxs[scan_id]:
                if frame_idx not in self.patch_anno[scan_id]:
                    continue
                data_item_dict = {}
                # 2D patch feature info
                patch_feature_path = self.features_path[scan_id][frame_idx]
                data_item_dict['patch_feature_path'] = patch_feature_path

                data_item_dict['frame_idx'] = frame_idx
                # 3D info
                data_item_dict['scan_id'] = scan_id
                data_item_dict['target_scan_id'] = target_scan_id
                data_items.append(data_item_dict)

        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:1]
        return data_items
    
    def dataItem2DataDict(self, data_item):
        scan_id = data_item['scan_id']
        frame_idx = data_item['frame_idx']
        patch_anno_frame = self.patch_anno[scan_id][frame_idx]
        obj_2D_patch_anno_flatten = patch_anno_frame.reshape(-1).astype(np.int32)
        # load 2D patch features
        patch_feature = np.load(data_item['patch_feature_path'])
        if patch_feature.ndim == 2:
            patch_feature = patch_feature.reshape(self.patch_h, 
                                self.patch_w , self.img_patch_feat_dim)
        
        data_dict = {
            'scan_id': scan_id,
            'target_scan_id': data_item['target_scan_id'], # for val&test split
            'frame_idx': frame_idx,
            'patch_features': patch_feature, 
            'obj_2D_patch_anno_flatten': obj_2D_patch_anno_flatten,
        }
        return data_dict
    
    def generateObjPatchAssociationDataDict(self, data_item, candidate_scans, sg_obj_idxs):
        target_scan_id = data_item['target_scan_id']
        if candidate_scans is None:
            candidate_scans_cur = []
        else:
            candidate_scans_cur = candidate_scans[target_scan_id]
        gt_2D_anno_flat = data_item['obj_2D_patch_anno_flatten']
        
        assoc_data_dict = self.generateObjPatchAssociationScan(
            target_scan_id, candidate_scans_cur, gt_2D_anno_flat, sg_obj_idxs)
        return assoc_data_dict
        
    def generateObjPatchAssociationScan(self, scan_id, candidate_scans, gt_2D_anno_flat, sg_obj_idxs):
        obj_3D_idx2info = {}
        obj_3D_id2idx_cur_scan = {} # for objs in current scene
        scans_sg_obj_idxs = [] # for current scene and other scenes
        candata_scan_obj_idxs = {}
        
        ## cur scan objs
        objs_ids_cur_scan = self.scene_graphs[scan_id]['obj_ids']
        idx = 0
        for obj_id in objs_ids_cur_scan:
            obj_3D_idx2info[idx] = self.obj_3D_anno[scan_id][obj_id]
            obj_3D_id2idx_cur_scan[obj_id] = idx
            scans_sg_obj_idxs.append(sg_obj_idxs[scan_id][obj_id])
            if scan_id not in candata_scan_obj_idxs:
                candata_scan_obj_idxs[scan_id] = []
            candata_scan_obj_idxs[scan_id].append(idx)
            idx += 1 
        candata_scan_obj_idxs[scan_id] = torch.Tensor(candata_scan_obj_idxs[scan_id]).long()
        ## other scans objs
        for cand_scan_id in candidate_scans:
            objs_ids_cand_scan = self.scene_graphs[cand_scan_id]['obj_ids']
            for obj_id in objs_ids_cand_scan:
                obj_3D_idx2info[idx] = self.obj_3D_anno[cand_scan_id][obj_id]
                scans_sg_obj_idxs.append(sg_obj_idxs[cand_scan_id][obj_id])
                if cand_scan_id not in candata_scan_obj_idxs:
                    candata_scan_obj_idxs[cand_scan_id] = []
                candata_scan_obj_idxs[cand_scan_id].append(idx)
                idx += 1
                
            candata_scan_obj_idxs[cand_scan_id] = torch.Tensor(
                candata_scan_obj_idxs[cand_scan_id]).long()
        ## to numpy
        scans_sg_obj_idxs = np.array(scans_sg_obj_idxs, dtype=np.int32)
        ## to torch
        scans_sg_obj_idxs = torch.from_numpy(scans_sg_obj_idxs).long()
        
        ## generate obj patch association
        ## From 2D to 3D, denote as e1i_matrix, e1j_matrix, e2j_matrix      
        ## e1i_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object pairs
        ## e2j_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object unpairs
        num_objs = idx
        e1i_matrix = np.zeros( (self.num_patch, num_objs), dtype=np.uint8)
        e2j_matrix = np.ones( (self.num_patch, num_objs), dtype=np.uint8)
        for patch_h_i in range(self.patch_h):
            patch_h_shift = patch_h_i*self.patch_w
            for patch_w_j in range(self.patch_w):
                obj_id = gt_2D_anno_flat[patch_h_shift + patch_w_j]
                if obj_id != self.undefined and (obj_id in obj_3D_id2idx_cur_scan):
                    obj_idx = obj_3D_id2idx_cur_scan[obj_id]
                    e1i_matrix[patch_h_shift+patch_w_j, obj_idx] = 1 # mark 2D-3D patch-object pairs
                    e2j_matrix[patch_h_shift+patch_w_j, obj_idx] = 0 # mark 2D-3D patch-object unpairs
        ## e1j_matrix, (num_patch, num_patch), mark unpaired patch-patch pair for image patches
        e1j_matrix = np.zeros( (self.num_patch, self.num_patch), dtype=np.uint8)
        for patch_h_i in range(self.patch_h):
            patch_h_shift = patch_h_i*self.patch_w
            for patch_w_j in range(self.patch_w):
                obj_id = gt_2D_anno_flat[patch_h_shift + patch_w_j]
                if obj_id != self.undefined and obj_id in obj_3D_id2idx_cur_scan:
                    e1j_matrix[patch_h_shift+patch_w_j, :] = np.logical_and(
                        gt_2D_anno_flat != self.undefined, gt_2D_anno_flat != obj_id
                    )
                else:
                     e1j_matrix[patch_h_shift+patch_w_j, :] = 1
        ## From 3D to 2D, denote as f1i_matrix, f1j_matrix, f2j_matrix
        ## f1i_matrix = e1i_matrix.T, thus skip
        ## f2j_matrix = e2j_matrix.T, thus skip
        ## f1j_matrix
        obj_cates = [obj_3D_idx2info[obj_idx][2] for obj_idx in range(len(obj_3D_idx2info))]
        obj_cates_arr = np.array(obj_cates)
        f1j_matrix = obj_cates_arr.reshape(1, -1) != obj_cates_arr.reshape(-1, 1)
                     
        assoc_data_dict = {
            'e1i_matrix': torch.from_numpy(e1i_matrix).float(),
            'e1j_matrix': torch.from_numpy(e1j_matrix).float(),
            'e2j_matrix': torch.from_numpy(e2j_matrix).float(),
            'f1j_matrix': torch.from_numpy(f1j_matrix).float(),
            'scans_sg_obj_idxs': scans_sg_obj_idxs,
            'candata_scan_obj_idxs': candata_scan_obj_idxs,
        }
        return assoc_data_dict
    
    def aggretateDataDicts(self, data_dict, key, mode):
        if mode == 'torch_cat':
            return torch.cat([data[key] for data in data_dict])
        elif mode == 'torch_stack':
            return torch.stack([data[key] for data in data_dict])
        elif mode == 'np_concat':
            return np.concatenate([data[key] for data in data_dict])
        elif mode == 'np_stack':
            return np.stack([data[key] for data in data_dict])
        else:
            raise NotImplementedError
    
    def collateBatchDicts(self, batch):
        scans_batch = [data['target_scan_id'] for data in batch]
        
        # sample candidate scenes for each scan
        if self.use_cross_scene:
            if self.split == 'train':
                candidate_scans, union_scans = self.sampleCandidateScenesForScans(scans_batch, self.num_scenes)
            else:
                candidate_scans = {}
                for scan_id in scans_batch:
                    candidate_scans[scan_id] = self.candidate_scans[scan_id]
                union_scans = list(set(scans_batch + [scan for scan_list in candidate_scans.values() for scan in scan_list]))
        else:
            candidate_scans, union_scans = None, scans_batch
                
        batch_size = len(batch)
        data_dict = {}
        data_dict['batch_size'] = batch_size
        # frame info 
        data_dict['scan_ids'] = np.stack([data['scan_id'] for data in batch])
        data_dict['target_scan_ids'] = np.stack([data['target_scan_id'] for data in batch])
        data_dict['frame_idxs'] = np.stack([data['frame_idx'] for data in batch])
        # 2D img info
        patch_features_batch = np.stack([data['patch_features'] for data in batch]) # (B, P_H, P_W, D)
        data_dict['patch_features'] = torch.from_numpy(patch_features_batch).float() # (B, H, W, C)
        data_dict['obj_2D_patch_anno_flatten_list'] = \
            [ torch.from_numpy(data['obj_2D_patch_anno_flatten']) for data in batch] # B - [N_P]

        # 3D scene graph info
        ## scene graph info
        scene_graph_scans = union_scans
        scene_graph_infos = [self.scene_graphs[scan_id] for scan_id in scene_graph_scans]
        scene_graphs_ = {}
        scans_size = len(scene_graph_infos)
        scene_graphs_['batch_size'] = scans_size
        scene_graphs_['scene_ids'] = self.aggretateDataDicts(scene_graph_infos, 'scene_ids', 'np_stack')
        scene_graphs_['obj_ids'] = self.aggretateDataDicts(scene_graph_infos, 'obj_ids', 'np_concat')
        scene_graphs_['tot_obj_pts'] = self.aggretateDataDicts(scene_graph_infos, 'tot_obj_pts', 'torch_cat')
        scene_graphs_['pcl_center'] = self.aggretateDataDicts(scene_graph_infos, 'pcl_center', 'np_stack')
        scene_graphs_['graph_per_obj_count'] = self.aggretateDataDicts(scene_graph_infos, 'graph_per_obj_count', 'np_stack')
        scene_graphs_['tot_obj_count'] = self.aggretateDataDicts(scene_graph_infos, 'tot_obj_count', 'np_stack')
        if 'attr' in self.sgaligner_modules:
            scene_graphs_['tot_bow_vec_object_attr_feats'] = \
                self.aggretateDataDicts(scene_graph_infos, 'tot_bow_vec_object_attr_feats', 'torch_cat').double()
        if 'rel' in self.sg_modules or 'gat' in self.sg_modules:
            scene_graphs_['graph_per_edge_count'] = self.aggretateDataDicts(scene_graph_infos, 'graph_per_edge_count', 'np_stack')
            scene_graphs_['tot_bow_vec_object_edge_feats'] = \
                self.aggretateDataDicts(scene_graph_infos, 'tot_bow_vec_object_edge_feats', 'torch_cat').double()
            scene_graphs_['tot_rel_pose'] = self.aggretateDataDicts(scene_graph_infos, 'tot_rel_pose', 'torch_cat').double()
            scene_graphs_['edges'] = self.aggretateDataDicts(scene_graph_infos, 'edges', 'torch_cat')
        ### 3D pcs data augmentation by elastic distortion
        if self.use_aug and self.split == 'train':
            num_obs = scene_graphs_['tot_obj_pts'].shape[1]
            pcs_flatten = scene_graphs_['tot_obj_pts'].reshape(-1, 3)
            pcs_distorted_flatten = self.elastic_distortion(pcs_flatten)
            scene_graphs_['tot_obj_pts'] = pcs_distorted_flatten.reshape(-1, num_obs, 3)
        ### img patch features 
        if 'img_patch' in self.sgaligner_modules:
            obj_img_patches = {}
            obj_count_ = 0
            for scan_idx, scan_id in enumerate(scene_graphs_['scene_ids']):
                scan_id = scan_id[0]
                
                obj_start_idx, obj_end_idx = obj_count_, obj_count_ + scene_graphs_['tot_obj_count'][scan_idx]
                obj_ids = scene_graphs_['obj_ids'][obj_start_idx: obj_end_idx]
                obj_img_patches_scan_tops = self.obj_img_patches_scan_tops[scan_id]
                obj_img_patches_scan = obj_img_patches_scan_tops['obj_visual_emb']
                obj_top_frames = obj_img_patches_scan_tops['obj_image_votes_topK']
                
                obj_img_patches[scan_id] = {}
                for obj_id in obj_ids:
                    if obj_id not in obj_top_frames:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                        obj_img_patches[scan_id][obj_id] = torch.from_numpy(obj_img_patch_embs).float()
                        continue
                    
                    obj_img_patch_embs_list = []
                    obj_frames = obj_top_frames[obj_id][:self.obj_topk] if len(obj_top_frames[obj_id]) >= self.obj_topk \
                        else obj_top_frames[obj_id]
                    for frame_idx in obj_frames:
                        if obj_img_patches_scan[obj_id][frame_idx] is not None:
                            embs_frame = obj_img_patches_scan[obj_id][frame_idx]
                            embs_frame = embs_frame.reshape(1, -1) if embs_frame.ndim == 1 else embs_frame
                            obj_img_patch_embs_list.append(embs_frame)
                        
                    if len(obj_img_patch_embs_list) == 0:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                    else:
                        obj_img_patch_embs = np.concatenate(obj_img_patch_embs_list, axis=0)
                        
                    obj_img_patches[scan_id][obj_id] = torch.from_numpy(obj_img_patch_embs).float()
                    
                obj_count_ += scene_graphs_['tot_obj_count'][scan_idx]
            scene_graphs_['obj_img_patches'] = obj_img_patches
        
        data_dict['scene_graphs'] = scene_graphs_
        ## obj info
        assoc_data_dict = []
        ### get sg obj idx 
        sg_obj_idxs = {}
        sg_obj_idxs_tensor = {}
        sg_obj_idx_start = 0
        for scan_idx, scan_id in enumerate(scene_graphs_['scene_ids']):
            scan_id = scan_id[0]
            sg_obj_idxs[scan_id] = {}
            objs_count = scene_graphs_['tot_obj_count'][scan_idx]
            sg_obj_idxs_tensor[scan_id] = torch.from_numpy(
                 scene_graphs_['obj_ids'][sg_obj_idx_start: sg_obj_idx_start+objs_count]).long()
            for sg_obj_idx in range(sg_obj_idx_start, sg_obj_idx_start+objs_count):
                obj_id = scene_graphs_['obj_ids'][sg_obj_idx]
                sg_obj_idxs[scan_id][obj_id] = sg_obj_idx
            sg_obj_idx_start += objs_count
        for data in batch:
            assoc_data_dict_curr = \
                self.generateObjPatchAssociationDataDict(data, candidate_scans, sg_obj_idxs)
            assoc_data_dict.append(assoc_data_dict_curr)
        data_dict['assoc_data_dict'] = assoc_data_dict
        data_dict['sg_obj_idxs'] = sg_obj_idxs
        data_dict['sg_obj_idxs_tensor'] = sg_obj_idxs_tensor
        data_dict['candidate_scans'] = candidate_scans
        if len(batch) > 0:
            return data_dict
        else:
            return None
    
    def __getitem__(self, idx):
        data_dict = self.dataItem2DataDict(self.data_items[idx])
        return data_dict
    
    def collate_fn(self, batch):
        return self.collateBatchDicts(batch)
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from datasets.loaders import get_val_dataloader
    from configs import config, update_config
    os.environ['Data_ROOT_DIR'] = '/home/yang/990Pro/scannet_seqs/data'
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week16/Test_PI_scannet_GTSeg/Test_PI_scannet_GTSeg.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    # train_dataloader, val_dataloader = get_train_val_data_loader(cfg, ScannetPatchObjDataset)
    val_dataset, val_dataloader = get_val_dataloader(cfg, ScannetPatchObjDataset)
    pbar = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
    for iteration, data_dict in pbar:
        pass
    
    breakpoint = None
    
    # scan3r_ds = ScannetPatchObjDataset(cfg, split='val')
    # batch_size = 16
    # for batch_i in  tqdm.tqdm(range(int(len(scan3r_ds)/batch_size))):
    #     batch = [scan3r_ds[i] for i in range(batch_i*batch_size, (batch_i+1)*batch_size)]
    #     data_batch = scan3r_ds.collate_fn(batch)