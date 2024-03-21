import os
import os.path as osp
from weakref import ref
from attr import assoc
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from yaml import scan

sys.path.append('..')
sys.path.append('../..')
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

class ElasticDistortion: # from torch-points3d
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a 
    voxel grid and then apply the distortion to the voxel grid.

    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self, apply_distorsion: bool = True, granularity: list = [0.2, 0.8], magnitude=[0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords).float()

    def __call__(self, pcs_pos):
        # coords = pcs_pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    pcs_pos = ElasticDistortion.elastic_distortion(pcs_pos, self._granularity[i], self._magnitude[i],)
        return pcs_pos

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude,
        )

class PatchObjectPairXTAESGIDataSet(data.Dataset):
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
        self.sgaligner_modules = cfg.sgaligner.modules
        
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
        # 3D_obj_embedding_dir
        self.scans_3Dobj_embeddings_dir = osp.join(self.scans_files_dir_mode, "embeddings")
        # 2D images
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        # 2D_patch_anno size
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        # 2D patch anno
        if self.img_rotate:
            self.patch_h = self.image_patch_w
            self.patch_w = self.image_patch_h
        else:
            self.patch_h = self.image_patch_h
            self.patch_w = self.image_patch_w
        self.step = self.cfg.data.img.img_step
        self.num_patch = self.patch_h * self.patch_w
    
        # scene_img_dir
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        
        # cross scenes cfg
        self.use_cross_scene = cfg.data.cross_scene.use_cross_scene
        self.num_scenes = cfg.data.cross_scene.num_scenes
        self.num_negative_samples = cfg.data.cross_scene.num_negative_samples
        self.use_tf_idf = cfg.data.cross_scene.use_tf_idf
        
        # if split is val, then use all object from other scenes as negative samples
        # if room_retrieval, then use load additional data items
        self.rescan = cfg.data.rescan
        self.room_retrieval = False
        if split == 'val' or split == 'test':
            self.num_negative_samples = -1
            self.room_retrieval = True
            self.room_retrieval_epsilon_th = cfg.val.room_retrieval.epsilon_th
            self.rescan = True
            self.step = 1
            
        
        # scans info
        self.temporal = cfg.data.temporal
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
            
        # load 2D image paths
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id, self.step)
        # load 2D frame poses
        self.image_poses = {}
        for scan_id in self.scan_ids:
            frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id)
            self.image_poses[scan_id] = scan3r.load_frame_poses(
                self.scans_scenes_dir, scan_id, frame_idxs, type='quat_trans')
            
        # load 2D patch features if use pre-calculated feature
        self.use_2D_feature = cfg.data.img_encoding.use_feature
        self.preload_2D_feature = cfg.data.img_encoding.preload_feature
        self.patch_feature_folder = osp.join(self.scans_files_dir, self.cfg.data.img_encoding.feature_dir)
        self.patch_features = {}
        self.patch_features_paths = {}
        if self.use_2D_feature:
            if self.preload_2D_feature:
                for scan_id in self.scan_ids:
                    self.patch_features[scan_id] = scan3r.load_patch_feature_scans(
                        self.data_root_dir, self.patch_feature_folder, scan_id, self.step)
            else:
                for scan_id in self.scan_ids:
                    self.patch_features_paths[scan_id] = osp.join(self.patch_feature_folder, "{}.pkl".format(scan_id))
                
        # # load 2D gt obj id annotation patch
        # self.gt_2D_anno_folder = osp.join(self.scans_files_dir, 'gt_projection/obj_id_pkl')
        # self.obj_2D_annos_path = {}
        # for scan_id in self.scan_ids:
        #     anno_2D_file = osp.join(self.gt_2D_anno_folder, "{}.pkl".format(scan_id))
        #     self.obj_2D_annos_path[scan_id] = anno_2D_file
        
        # load patch anno
        self.patch_anno = {}
        self.patch_anno_folder = osp.join(self.scans_files_dir, 'patch_anno/patch_anno_16_9')
        for scan_id in self.scan_ids:
            self.patch_anno[scan_id] = common.load_pkl_data(osp.join(self.patch_anno_folder, "{}.pkl".format(scan_id)))
        
        # load 3D scene graph information
        self.load3DSceneGraphs()
        ## load obj_visual features
        self.img_patch_feat_dim = self.cfg.sgaligner.model.img_patch_feat_dim
        obj_img_patch_name = self.cfg.data.scene_graph.obj_img_patch
        self.obj_patch_num = self.cfg.data.scene_graph.obj_patch_num
        self.obj_topk = self.cfg.data.scene_graph.obj_topk
        self.use_pos_enc = self.cfg.sgaligner.use_pos_enc
        self.obj_img_patches_scan_tops = {}
        if 'img_patch' in self.sgaligner_modules:
            for scan_id in self.all_scans_split:
                obj_visual_file = osp.join(self.scans_files_dir, obj_img_patch_name, scan_id+'.pkl')
                self.obj_img_patches_scan_tops[scan_id] = common.load_pkl_data(obj_visual_file)
                
        # set data augmentation
        self.use_aug = cfg.train.data_aug.use_aug
        ## 2D image
        self.img_rot = cfg.train.data_aug.img.rotation
        self.img_Hor_flip = cfg.train.data_aug.img.horizontal_flip
        self.img_Ver_flip = cfg.train.data_aug.img.vertical_flip
        self.img_jitter = cfg.train.data_aug.img.color
        self.trans_2D = A.Compose(
            transforms=[
                A.VerticalFlip(p=self.img_Ver_flip),
                A.HorizontalFlip(p=self.img_Hor_flip),
                A.Rotate(limit=self.img_rot, p=0.8, 
                        interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_CONSTANT, value=0)]
        )
        color_jitter = self.img_jitter
        self.brightness_2D = A.ColorJitter(
            brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter)
        
        ## 3D obj TODO
        self.elastic_distortion = ElasticDistortion(
            apply_distorsion=cfg.train.data_aug.use_aug_3D,
            granularity=cfg.train.data_aug.pcs.granularity,
            magnitude=cfg.train.data_aug.pcs.magnitude,
        )
            
        # fix candidate scan for val&test split for room retrieval
        if self.split == 'val' or self.split == 'test':
            self.candidate_scans = {}
            for scan_id in self.scan_ids:
                self.candidate_scans[scan_id] = scan3r.sampleCandidateScenesForEachScan(
                    scan_id, self.scan_ids, self.refscans2scans, self.scans2refscans, self.num_scenes)
            
            
        # generate data items given multiple scans
        self.data_items = self.generateDataItems()

    def load3DSceneGraphs(self):
        # load scene graph
        if self.split == 'train':
            self.pc_resolution = self.cfg.sgaligner.train.pc_res
        else:
            self.pc_resolution = self.cfg.sgaligner.val.pc_res
            
        rel_dim = self.cfg.sgaligner.model.rel_dim
        if rel_dim == 41:
            sg_filename = "data"
        elif rel_dim == 9:
            sg_filename = "data_rel9"
        else:
            raise ValueError("Invalid rel_dim")
        
        self.scene_graphs = {}
        for scan_id in self.all_scans_split:
            # Centering
            points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(scan_id)), obj_ids = None)
            pcl_center = np.mean(points, axis=0)
            # scene graph info
            scene_graph_dict = common.load_pkl_data(osp.join(self.scans_files_dir_mode, '{}/{}.pkl'.format(sg_filename, scan_id)))
            object_ids = scene_graph_dict['objects_id']
            global_object_ids = scene_graph_dict['objects_cat']
            edges = scene_graph_dict['edges']
            object_points = scene_graph_dict['obj_points'][self.pc_resolution] - pcl_center
            # load data to tensor
            object_points = torch.from_numpy(object_points).type(torch.FloatTensor)
            edges = torch.from_numpy(edges)
            if not self.use_predicted:
                bow_vec_obj_attr_feats = torch.from_numpy(scene_graph_dict['bow_vec_object_attr_feats'])
            else:
                bow_vec_obj_attr_feats = torch.zeros(object_points.shape[0], rel_dim)
            bow_vec_obj_edge_feats = torch.from_numpy(scene_graph_dict['bow_vec_object_edge_feats'])
            rel_pose = torch.from_numpy(scene_graph_dict['rel_trans'])
            # aggreate data 
            data_dict = {} 
            data_dict['obj_ids'] = object_ids
            data_dict['tot_obj_pts'] = object_points
            data_dict['graph_per_obj_count'] = np.array([object_points.shape[0]])
            data_dict['graph_per_edge_count'] = np.array([edges.shape[0]])
            data_dict['tot_obj_count'] = object_points.shape[0]
            data_dict['tot_bow_vec_object_attr_feats'] = bow_vec_obj_attr_feats
            data_dict['tot_bow_vec_object_edge_feats'] = bow_vec_obj_edge_feats
            data_dict['tot_rel_pose'] = rel_pose
            data_dict['edges'] = edges    
            data_dict['global_obj_ids'] = global_object_ids
            data_dict['scene_ids'] = [scan_id]        
            data_dict['pcl_center'] = pcl_center
            # get scene graph
            self.scene_graphs[scan_id] = data_dict
            
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
        
        ## category id to name
        self.obj_nyu40_id2name = common.idx2name(osp.join(self.scans_files_dir, 'scannet40_classes.txt'))


    def sampleCandidateScenesForEachScan(self, scan_id, num_scenes):
        candidate_scans = []
        scans_same_scene = self.refscans2scans[self.scans2refscans[scan_id]]
        # sample other scenes
        for scan in self.all_scans_split:
            if scan not in scans_same_scene:
                candidate_scans.append(scan)
        sampled_scans = random.sample(candidate_scans, num_scenes)
        return sampled_scans
    
    # def sampleCandidateScenesForScans(self, scan_ids, num_scenes):
    #     candidate_scans = {}
        
    #     # ref scans of input scans
    #     ref_scans = [self.scans2refscans[scan_id] for scan_id in scan_ids]
    #     ref_scans = list(set(ref_scans))
    #     num_ref_scans = len(ref_scans)
        
    #     if num_ref_scans > num_scenes: # if enough ref scans, no need to sample other scenes
    #         for scan_id in scan_ids:
    #             candidate_scan_pool = [scan for scan in scan_ids if scan not in self.refscans2scans[self.scans2refscans[scan_id]]]
    #             candidate_scan_pool = list(set(candidate_scan_pool))
    #             candidate_scans[scan_id] = random.sample(candidate_scan_pool, num_scenes)
    #     else: # if not enough ref scans, sample other additional scenes
    #         num_scans_to_be_sampled = num_scenes - num_ref_scans + 1
    #         additional_candidate_sample_pool = [scan for scan in self.all_scans_split if self.scans2refscans[scan] not in ref_scans]
    #         additional_candidates = random.sample(additional_candidate_sample_pool, num_scans_to_be_sampled)
    #         for scan_id in scan_ids:
    #             # first get scans in the batch
    #             candidate_scan_pool = [scan for scan in scan_ids if self.scans2refscans[scan] != self.scans2refscans[scan_id]]
    #             candidate_scan_pool = list(set(candidate_scan_pool))
    #             num_scans_to_be_sampled_curr = num_scenes - len(candidate_scan_pool)
    #             candidate_scans_curr = candidate_scan_pool + random.sample(additional_candidates, num_scans_to_be_sampled_curr)
    #             candidate_scans[scan_id] = list(set(candidate_scans_curr))
    #     candidate_scans_all = list(set([scan for scan_list in candidate_scans.values() for scan in scan_list]))
    #     union_scans = list(set(scan_ids + candidate_scans_all))
    #     return candidate_scans, union_scans
    
    def sampleCandidateScenesForScans(self, scan_ids, num_scenes):
        candidate_scans = {}
        # ref scans of input scans
        ref_scans = [self.scans2refscans[scan_id] for scan_id in scan_ids]
        ref_scans = list(set(ref_scans))
        num_ref_scans = len(ref_scans)
        num_scans_to_be_sampled = num_scenes
        additional_candidate_sample_pool = [scan for scan in self.all_scans_split if self.scans2refscans[scan] not in ref_scans]
        additional_candidates = random.sample(additional_candidate_sample_pool, num_scans_to_be_sampled)
        for scan_id in scan_ids:
            candidate_scans[scan_id] = list(set(additional_candidates))      
        candidate_scans_all = list(set([scan for scan_list in candidate_scans.values() for scan in scan_list]))
        union_scans = list(set(scan_ids + candidate_scans_all))
        return candidate_scans, union_scans
    
    # sample cross time for each data item
    def sampleScanCrossTime(self, scan_id):
        candidate_scans = []
        ref_scan = self.scans2refscans[scan_id]
        for scan in self.refscans2scans[ref_scan]:
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
            # iterate over images
            image_paths = self.image_paths[scan_id]
            for frame_idx in image_paths:
                data_item_dict = {}
                # 2D info
                if self.use_2D_feature:
                    if self.preload_2D_feature:
                        data_item_dict['patch_features'] = self.patch_features[scan_id][frame_idx]
                    else:
                        data_item_dict['patch_features_path'] = self.patch_features_paths[scan_id]
                else:
                    data_item_dict['img_path'] = image_paths[frame_idx]
                data_item_dict['frame_idx'] = frame_idx
                # 3D info
                data_item_dict['scan_id'] = scan_id
                data_items.append(data_item_dict)

        # if debug with single scan
        if self.cfg.mode == "debug_few_scan":
            return data_items[:50]
        return data_items
    
    def dataItem2DataDict(self, data_item, temporal=False):
        data_dict = {}
        
        # scan id of data point
        scan_id = data_item['scan_id']
        # sample over time if temporal
        if temporal:
            data_dict['scan_id_temporal'] = self.sampleScanCrossTime(scan_id)
            
        # 2D data
        frame_idx = data_item['frame_idx']
        
        ## 2D path features
        if self.use_2D_feature:
            if self.preload_2D_feature:
                patch_features = data_item['patch_features']
            else:
                patch_features = common.load_pkl_data(data_item['patch_features_path'])[frame_idx]
            if patch_features.ndim == 2:
                patch_features = patch_features.reshape(self.patch_h, self.patch_w , self.img_patch_feat_dim)
        else:
            # img data
            img_path = data_item['img_path']
            img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) # type: ignore
            img = cv2.resize(img, (self.image_resize_w, self.image_resize_h),  # type: ignore
                            interpolation=cv2.INTER_LINEAR) # type: ignore
            if self.img_rotate:
                img = img.transpose(1, 0, 2)
                img = np.flip(img, 1)
            ## 2D data augmentation
            if self.use_aug and self.split == 'train':
                augments_2D = self.trans_2D(image=img, mask=obj_2D_anno)
                img = augments_2D['image']
                obj_2D_anno = augments_2D['mask']
                img = self.brightness_2D(image=img)['image']
                
        ## patch anno
        patch_anno_frame = self.patch_anno[scan_id][frame_idx]
        if self.img_rotate:
            patch_anno_frame = patch_anno_frame.transpose(1, 0)
            patch_anno_frame = np.flip(patch_anno_frame, 1)
        obj_2D_patch_anno_flatten = patch_anno_frame.reshape(-1)
            
        # frame info
        data_dict['scan_id'] = scan_id
        data_dict['frame_idx'] = frame_idx
        data_dict['obj_2D_patch_anno_flatten'] = obj_2D_patch_anno_flatten
        if self.use_2D_feature:
            data_dict['patch_features'] = patch_features
        else:
            data_dict['image'] = img
            if self.cfg.data.img_encoding.record_feature:
                data_dict['patch_features_path'] = self.patch_features_paths[scan_id][frame_idx]
        return data_dict
    
    def generateObjPatchAssociationDataDict(self, data_item, candidate_scans, sg_obj_idxs):
        scan_id = data_item['scan_id']
        if candidate_scans is None:
            candidate_scans_cur = []
        else:
            candidate_scans_cur = candidate_scans[scan_id]
        gt_2D_anno_flat = data_item['obj_2D_patch_anno_flatten']
        assoc_data_dict = self.generateObjPatchAssociationScan(scan_id, candidate_scans_cur, gt_2D_anno_flat, sg_obj_idxs)
        
        # temporal 
        if self.temporal:
            scan_id_temporal = data_item['scan_id_temporal']
            assoc_data_dict_temporal = self.generateObjPatchAssociationScan(
                scan_id_temporal, candidate_scans_cur, gt_2D_anno_flat, sg_obj_idxs)
            return assoc_data_dict, assoc_data_dict_temporal
        else:
            return assoc_data_dict, None
        
    def generateObjPatchAssociationScan(self, scan_id, candidate_scans, gt_2D_anno_flat, sg_obj_idxs):
        obj_3D_idx2info = {} # for objs in current scene and other scenes
        obj_3D_id2idx_cur_scan = {} # for objs in current scene
        scans_sg_obj_idxs = [] # for current scene and other scenes
        candata_scan_obj_idxs = {}
        
        # for tf_idf
        all_candi_scans = [scan_id] + candidate_scans
        N_scenes = 1 + len(candidate_scans)
        n_scenes_per_sem = {} # number of scans containing certain semantic category
        n_words_per_scene = {candi_scan_id: 0 for candi_scan_id in all_candi_scans} # number of words in each scene
        n_word_scene = {candi_scan_id: {} for candi_scan_id in all_candi_scans} # number of words for each semantic in each scene
        reweight_matrix_scans = {candi_scan_id: None for candi_scan_id in all_candi_scans}
        cadidate_scans_semantic_ids = []
        
        ## cur scan objs
        objs_ids_cur_scan = self.scene_graphs[scan_id]['obj_ids']
        idx = 0
        for obj_id in objs_ids_cur_scan:
            obj_3D_idx2info[idx] = self.obj_3D_anno[scan_id][obj_id]
            obj_3D_id2idx_cur_scan[obj_id] = idx
            scans_sg_obj_idxs.append(sg_obj_idxs[scan_id][obj_id])
            cadidate_scans_semantic_ids.append(self.obj_3D_anno[scan_id][obj_id][2])
            if scan_id not in candata_scan_obj_idxs:
                candata_scan_obj_idxs[scan_id] = []
            candata_scan_obj_idxs[scan_id].append(idx)
            idx += 1 
            
            if self.use_tf_idf:
                obj_nyu_category = self.obj_3D_anno[scan_id][obj_id][2]
                if obj_nyu_category not in n_word_scene[scan_id]:
                    n_word_scene[scan_id][obj_nyu_category] = 0
                n_word_scene[scan_id][obj_nyu_category] += 1
                n_words_per_scene[scan_id] += 1
                if obj_nyu_category not in n_scenes_per_sem:
                    n_scenes_per_sem[obj_nyu_category] = set()
                n_scenes_per_sem[obj_nyu_category].add(scan_id)
        ## other scans objs
        for cand_scan_id in candidate_scans:
            objs_ids_cand_scan = self.scene_graphs[cand_scan_id]['obj_ids']
            for obj_id in objs_ids_cand_scan:
                obj_3D_idx2info[idx] = self.obj_3D_anno[cand_scan_id][obj_id]
                scans_sg_obj_idxs.append(sg_obj_idxs[cand_scan_id][obj_id])
                cadidate_scans_semantic_ids.append(
                    self.obj_3D_anno[cand_scan_id][obj_id][2])
                if cand_scan_id not in candata_scan_obj_idxs:
                    candata_scan_obj_idxs[cand_scan_id] = []
                candata_scan_obj_idxs[cand_scan_id].append(idx)
                idx += 1
                
                if self.use_tf_idf:
                    obj_nyu_category = self.obj_3D_anno[cand_scan_id][obj_id][2]
                    if obj_nyu_category not in n_word_scene[cand_scan_id]:
                        n_word_scene[cand_scan_id][obj_nyu_category] = 0
                    n_word_scene[cand_scan_id][obj_nyu_category] += 1
                    n_words_per_scene[cand_scan_id] += 1
                    if obj_nyu_category not in n_scenes_per_sem:
                        n_scenes_per_sem[obj_nyu_category] = set()
                    n_scenes_per_sem[obj_nyu_category].add(cand_scan_id)
                
            candata_scan_obj_idxs[cand_scan_id] = torch.Tensor(
                candata_scan_obj_idxs[cand_scan_id]).long()
        candata_scan_obj_idxs[scan_id] = torch.Tensor(candata_scan_obj_idxs[scan_id]).long()
        ## to numpy
        scans_sg_obj_idxs = np.array(scans_sg_obj_idxs, dtype=np.int32)
        cadidate_scans_semantic_ids = np.array(cadidate_scans_semantic_ids, dtype=np.int32)
        ## to torch
        scans_sg_obj_idxs = torch.from_numpy(scans_sg_obj_idxs).long()
        cadidate_scans_semantic_ids = torch.from_numpy(cadidate_scans_semantic_ids).long()
        ## calculate tf_idf reweight matrix for each object in each scene
        if self.use_tf_idf:
            for cand_scan_id in all_candi_scans:
                objs_ids_cand_scan = self.scene_graphs[cand_scan_id]['obj_ids']
                reweight_matrix_scans[cand_scan_id] = torch.zeros(len(objs_ids_cand_scan))
                obj_idx = 0
                for obj_id in objs_ids_cand_scan:
                    obj_nyu_category = self.obj_3D_anno[cand_scan_id][obj_id][2]
                    n_word_scene_obj = n_word_scene[cand_scan_id][obj_nyu_category]
                    n_words_curr_scene = n_words_per_scene[cand_scan_id]
                    n_scenes_per_sem_obj = len(n_scenes_per_sem[obj_nyu_category])
                    idf = np.log(N_scenes / n_scenes_per_sem_obj)
                    tf_idf =n_word_scene_obj / n_words_curr_scene * idf
                    reweight_matrix_scans[cand_scan_id][obj_idx] =  max( tf_idf, 1e-3)
                    obj_idx += 1
        ## generate obj patch association
        ## From 2D to 3D, denote as e1i_matrix, e1j_matrix, e2j_matrix      
        ## e1i_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object pairs
        ## e2j_matrix,(num_patch, num_3D_obj), record 2D-3D patch-object unpairs
        num_objs = idx
        gt_patch_cates = np.zeros(self.num_patch, dtype=np.uint8)
        e1i_matrix = np.zeros( (self.num_patch, num_objs), dtype=np.uint8)
        e2j_matrix = np.ones( (self.num_patch, num_objs), dtype=np.uint8)
        for patch_h_i in range(self.patch_h):
            patch_h_shift = patch_h_i*self.patch_w
            for patch_w_j in range(self.patch_w):
                patch_idx = patch_h_shift + patch_w_j
                obj_id = gt_2D_anno_flat[patch_idx]
                if obj_id != self.undefined and (obj_id in obj_3D_id2idx_cur_scan):
                    obj_idx = obj_3D_id2idx_cur_scan[obj_id]
                    e1i_matrix[patch_h_shift+patch_w_j, obj_idx] = 1 # mark 2D-3D patch-object pairs
                    e2j_matrix[patch_h_shift+patch_w_j, obj_idx] = 0 # mark 2D-3D patch-object unpairs
                    gt_patch_cates[patch_idx] = self.obj_3D_anno[scan_id][obj_id][2]
                else:
                    gt_patch_cates[patch_idx] = self.undefined
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
            'gt_patch_cates': gt_patch_cates,
            'scans_sg_obj_idxs': scans_sg_obj_idxs,
            'cadidate_scans_semantic_ids': cadidate_scans_semantic_ids,
            'candata_scan_obj_idxs': candata_scan_obj_idxs,
            'reweight_matrix_scans': reweight_matrix_scans,
            'n_scenes_per_sem': n_scenes_per_sem,
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
        scans_batch = [data['scan_id'] for data in batch]
        
        # sample candidate scenes for each scan
        if self.use_cross_scene:
            if self.split == 'train':
                candidate_scans, union_scans = self.sampleCandidateScenesForScans(scans_batch, self.num_scenes)
            else:
                candidate_scans = {}
                for scan_id in scans_batch:
                    candidate_scans[scan_id] = self.candidate_scans[scan_id]
                union_scans = list(set(scans_batch + [scan for scan_list in candidate_scans.values() for scan in scan_list]))
            # candidate_scans, union_scans = self.sampleCandidateScenesForScans(scans_batch, self.num_scenes)
        else:
            candidate_scans, union_scans = None, scans_batch
        
        batch_size = len(batch)
        data_dict = {}
        data_dict['batch_size'] = batch_size
        data_dict['temporal'] = self.temporal
        # frame info 
        data_dict['scan_ids'] = np.stack([data['scan_id'] for data in batch])
        if self.temporal:
            data_dict['scan_ids_temp'] = np.stack([data['scan_id_temporal'] for data in batch])
        data_dict['frame_idxs'] = np.stack([data['frame_idx'] for data in batch])
        # 2D img info
        if self.use_2D_feature:
            patch_features_batch = np.stack([data['patch_features'] for data in batch]) # (B, P_H, P_W, D)
            data_dict['patch_features'] = torch.from_numpy(patch_features_batch).float() # (B, H, W, C)
        else:
            images_batch = np.stack([data['image'] for data in batch])
            data_dict['images'] = torch.from_numpy(images_batch).float() # (B, H, W, C)
            if self.cfg.data.img_encoding.record_feature:
                data_dict['patch_features_paths'] = [data['patch_features_path'] for data in batch]
        data_dict['obj_2D_patch_anno_flatten_list'] = \
            [ torch.from_numpy(data['obj_2D_patch_anno_flatten']) for data in batch] # B - [N_P]
        # 3D scene graph info
        ## scene graph info
        ### include temporal scans 
        if self.temporal:
            scene_graph_scans = list(set(union_scans + [data['scan_id_temporal'] for data in batch]))
        else:
            scene_graph_scans = union_scans
        scene_graph_infos = [self.scene_graphs[scan_id] for scan_id in scene_graph_scans]
        scene_graphs_ = {}
        scans_size = len(scene_graph_infos)
        scene_graphs_['batch_size'] = scans_size
        scene_graphs_['obj_ids'] = self.aggretateDataDicts(scene_graph_infos, 'obj_ids', 'np_concat')
        scene_graphs_['tot_obj_pts'] = self.aggretateDataDicts(scene_graph_infos, 'tot_obj_pts', 'torch_cat')
        scene_graphs_['graph_per_obj_count'] = self.aggretateDataDicts(scene_graph_infos, 'graph_per_obj_count', 'np_stack')
        scene_graphs_['graph_per_edge_count'] = self.aggretateDataDicts(scene_graph_infos, 'graph_per_edge_count', 'np_stack')
        scene_graphs_['tot_obj_count'] = self.aggretateDataDicts(scene_graph_infos, 'tot_obj_count', 'np_stack')
        scene_graphs_['tot_bow_vec_object_attr_feats'] = \
            self.aggretateDataDicts(scene_graph_infos, 'tot_bow_vec_object_attr_feats', 'torch_cat').double()
        scene_graphs_['tot_bow_vec_object_edge_feats'] = \
            self.aggretateDataDicts(scene_graph_infos, 'tot_bow_vec_object_edge_feats', 'torch_cat').double()
        scene_graphs_['tot_rel_pose'] = self.aggretateDataDicts(scene_graph_infos, 'tot_rel_pose', 'torch_cat').double()
        scene_graphs_['edges'] = self.aggretateDataDicts(scene_graph_infos, 'edges', 'torch_cat')
        scene_graphs_['global_obj_ids'] = self.aggretateDataDicts(scene_graph_infos, 'global_obj_ids', 'np_concat')
        scene_graphs_['scene_ids'] = self.aggretateDataDicts(scene_graph_infos, 'scene_ids', 'np_stack')
        scene_graphs_['pcl_center'] = self.aggretateDataDicts(scene_graph_infos, 'pcl_center', 'np_stack')
        ### 3D pcs data augmentation by elastic distortion
        if self.use_aug and self.split == 'train':
            num_obs = scene_graphs_['tot_obj_pts'].shape[1]
            pcs_flatten = scene_graphs_['tot_obj_pts'].reshape(-1, 3)
            pcs_distorted_flatten = self.elastic_distortion(pcs_flatten)
            scene_graphs_['tot_obj_pts'] = pcs_distorted_flatten.reshape(-1, num_obs, 3)
        ### img patch features 
        if 'img_patch' in self.sgaligner_modules:
            obj_img_patches = {}
            obj_img_poses = {}
            obj_count_ = 0
            for scan_idx, scan_id in enumerate(scene_graphs_['scene_ids']):
                scan_id = scan_id[0]
                
                obj_start_idx, obj_end_idx = obj_count_, obj_count_ + scene_graphs_['tot_obj_count'][scan_idx]
                obj_ids = scene_graphs_['obj_ids'][obj_start_idx: obj_end_idx]
                obj_img_patches_scan_tops = self.obj_img_patches_scan_tops[scan_id]
                obj_img_patches_scan = obj_img_patches_scan_tops['obj_visual_emb']
                obj_top_frames = obj_img_patches_scan_tops['obj_image_votes_topK']
                
                obj_img_patches[scan_id] = {}
                obj_img_poses[scan_id] = {}
                for obj_id in obj_ids:
                    if obj_id not in obj_top_frames:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                        obj_img_patches[scan_id][obj_id] = torch.from_numpy(obj_img_patch_embs).float()
                        if self.use_pos_enc:
                            identity_pos = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(1, -1)
                            obj_img_poses[scan_id][obj_id] = torch.from_numpy(identity_pos).float()
                        continue
                    
                    obj_img_patch_embs_list = []
                    obj_img_poses_list = []
                    obj_frames = obj_top_frames[obj_id][:self.obj_topk] if len(obj_top_frames[obj_id]) >= self.obj_topk \
                        else obj_top_frames[obj_id]
                    for frame_idx in obj_frames:
                        if obj_img_patches_scan[obj_id][frame_idx] is not None:
                            embs_frame = obj_img_patches_scan[obj_id][frame_idx]
                            embs_frame = embs_frame.reshape(1, -1) if embs_frame.ndim == 1 else embs_frame
                            obj_img_patch_embs_list.append(embs_frame)
                            if self.use_pos_enc:
                                obj_img_poses_list.append(self.image_poses[scan_id][frame_idx])
                        
                    if len(obj_img_patch_embs_list) == 0:
                        obj_img_patch_embs = np.zeros((1, self.img_patch_feat_dim))
                        if self.use_pos_enc:
                            obj_img_poses_arr = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(1, -1)
                    else:
                        obj_img_patch_embs = np.concatenate(obj_img_patch_embs_list, axis=0)
                        if self.use_pos_enc:
                            obj_img_poses_arr = np.stack(obj_img_poses_list, axis=0)
                        
                    obj_img_patches[scan_id][obj_id] = torch.from_numpy(obj_img_patch_embs).float()
                    if self.use_pos_enc:
                        obj_img_poses[scan_id][obj_id] = torch.from_numpy(obj_img_poses_arr).float()
                    
                obj_count_ += scene_graphs_['tot_obj_count'][scan_idx]
            scene_graphs_['obj_img_patches'] = obj_img_patches
            if self.use_pos_enc:
                scene_graphs_['obj_img_poses'] = obj_img_poses
        
        data_dict['scene_graphs'] = scene_graphs_
        
        ## obj info
        assoc_data_dict, assoc_data_dict_temporal = [], []
        ### get sg obj idx 
        sg_obj_idxs = {}
        sg_obj_idxs_tensor = {}
        sg_obj_idx_start = 0
        for scan_idx, scan_id in enumerate(scene_graphs_['scene_ids']):
            scan_id = scan_id[0]
            sg_obj_idxs[scan_id] = {}
            objs_count = scene_graphs_['tot_obj_count'][scan_idx]
            # sg_obj_idxs_tensor[scan_id] = torch.from_numpy(
            #      scene_graphs_['obj_ids'][sg_obj_idx_start: sg_obj_idx_start+objs_count]).long()
            sg_obj_idxs_tensor[scan_id] = torch.from_numpy(
                 np.arange(sg_obj_idx_start, sg_obj_idx_start+objs_count)).long()
            for sg_obj_idx in range(sg_obj_idx_start, sg_obj_idx_start+objs_count):
                obj_id = scene_graphs_['obj_ids'][sg_obj_idx]
                sg_obj_idxs[scan_id][obj_id] = sg_obj_idx
            sg_obj_idx_start += objs_count
        for data in batch:
            assoc_data_dict_curr, assoc_data_dict_temporal_curr = \
                self.generateObjPatchAssociationDataDict(data, candidate_scans, sg_obj_idxs)
            assoc_data_dict.append(assoc_data_dict_curr)
            assoc_data_dict_temporal.append(assoc_data_dict_temporal_curr)
        data_dict['assoc_data_dict'] = assoc_data_dict
        data_dict['assoc_data_dict_temp'] = assoc_data_dict_temporal
        data_dict['sg_obj_idxs'] = sg_obj_idxs
        data_dict['sg_obj_idxs_tensor'] = sg_obj_idxs_tensor
        data_dict['candidate_scans'] = candidate_scans
        if len(batch) > 0:
            return data_dict
        else:
            return None
    
    def __getitem__(self, idx):
        data_item = self.data_items[idx]
        data_dict = self.dataItem2DataDict(data_item, self.temporal)
        return data_dict
    
    def collate_fn(self, batch):
        return self.collateBatchDicts(batch)
        
    def __len__(self):
        return len(self.data_items)
    
if __name__ == '__main__':
    # TODO  check the correctness of dataset 
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from datasets.loaders import get_train_val_data_loader, get_val_dataloader
    from configs import config, update_config
    cfg_file = "/home/yang/big_ssd/Scan3R/VLSG/implementation/week16/DebugImgPosEnc/DebugImgPosEnc.yaml"
    cfg = update_config(config, cfg_file, ensure_dir = False)
    # train_dataloader, val_dataloader = get_train_val_data_loader(cfg, PatchObjectPairXTAESGIDataSet)
    val_dataset, val_dataloader = get_val_dataloader(cfg, PatchObjectPairXTAESGIDataSet)
    pbar = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
    for iteration, data_dict in pbar:
        pass
    
    # scan3r_ds = PatchObjectPairXTAESGIDataSet(cfg, split='val')
    # batch_size = 16
    # for batch_i in  tqdm.tqdm(range(int(len(scan3r_ds)/batch_size))):
    #     batch = [scan3r_ds[i] for i in range(batch_i*batch_size, (batch_i+1)*batch_size)]
    #     data_batch = scan3r_ds.collate_fn(batch)