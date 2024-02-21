import argparse
import os 
import os.path as osp
from re import T
import time
from tracemalloc import start
import comm
from matplotlib import patches
import numpy as np 
import sys
import subprocess
import tqdm

from requests import patch
from sympy import N
from yaml import scan

src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ws_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(ws_dir)
# utils
from utils import common
from utils import torch_util
# from utils import visualisation
# config
from configs import update_config_room_retrival, config
# tester
from engine.single_tester import SingleTester
from utils.summary_board import SummaryBoard

# from models.GCVit.models import gc_vit
from models.patch_SGIE_aligner import PatchSGIEAligner
# dataset
from datasets.loaders import get_test_dataloader, get_val_dataloader
from datasets.scan3r_openmask3d import Scan3rOpen3DDataset
from datasets.scannet_openmask3d import ScannetOpen3DDataset

# use PathObjAligner for room retrieval
class RoomRetrivalScore():
    def __init__(self, cfg):
        
        # cfg
        self.cfg = cfg 
        self.method_name = cfg.val.room_retrieval.method_name
        
        # dataloader
        if self.cfg.data.name == "Scan3R":
            dataset = Scan3rOpen3DDataset
        elif self.cfg.data.name == "Scannet":
            dataset = ScannetOpen3DDataset
        else:
            raise ValueError(f"Dataset {self.cfg.data.name} not supported.")
            
        start_time = time.time()
        val_dataset, val_data_loader = get_val_dataloader(cfg, Dataset = dataset)
        test_dataset, test_data_loader = get_test_dataloader(cfg, Dataset = dataset)
        # register dataloader
        self.val_data_loader = val_data_loader
        self.val_dataset = val_dataset
        self.test_data_loader = test_data_loader
        self.test_dataset = test_dataset
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        
        # model
        self.registerPatchObjectAlignerFromCfg(cfg)
        
        # results
        self.val_room_retrieval_summary = SummaryBoard(adaptive=True)
        self.test_room_retrieval_summary = SummaryBoard(adaptive=True)
        self.val_room_retrieval_record = {}
        self.test_room_retrieval_record = {}
        
        # files
        self.output_dir = osp.join(cfg.output_dir, self.method_name)
        common.ensure_dir(self.output_dir)

    def registerPatchObjectAlignerFromCfg(self, cfg):
        pass

    def model_forward(self, data_dict):
        pass 
    
    def room_retrieval_dict(self, data_dict, dataset, room_retrieval_record, record_retrieval = False):
        
        # room retrieval with scan point cloud
        batch_size = data_dict['batch_size']
        top_k_list = [1,3,5]
        top_k_recall_temporal = {"R@{}_T_S".format(k): 0. for k in top_k_list}
        top_k_recall_non_temporal = {"R@{}_NT_S".format(k): 0. for k in top_k_list}
        retrieval_time_temporal = 0.
        retrieval_time_non_temporal = 0.
        img_forward_time = 0.
        matched_obj_idxs, matched_obj_idxs_temp = None, None
        
        scans_objs_embeds = data_dict['obj_3D_embeddings']
        
        for batch_i in range(batch_size):
            patch_features = data_dict['img_features'][batch_i]
            patch_w, patch_h, feat_dim = patch_features.shape
            patch_features = patch_features.reshape(-1, feat_dim)
            # non-temporal
            room_score_scans_NT = {}
            target_scan_id = data_dict['scan_ids'][batch_i]
            
            ## start room retrieval in cpu
            candidate_scan_ids = data_dict['candidate_scan_ids_list'][batch_i]
            start_time = time.time()
            ## normalize patch features
            scans_to_match = list(set(candidate_scan_ids + [target_scan_id]))
            patch_features_norm = patch_features / np.linalg.norm(patch_features, axis=1, keepdims=True)
            for candidate_scan_id in scans_to_match:
                candidate_obj_embeds = scans_objs_embeds[candidate_scan_id]
                patch_obj_sim = patch_features_norm@candidate_obj_embeds.T
                matched_candidate_obj_sim = np.max(patch_obj_sim, axis=1)
                room_score_scans_NT[candidate_scan_id] = matched_candidate_obj_sim.sum()
            room_sorted_by_scores_NT =  [item[0] for item in sorted(room_score_scans_NT.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if target_scan_id in room_sorted_by_scores_NT[:k]:
                    top_k_recall_non_temporal["R@{}_NT_S".format(k)] += 1
            retrieval_time_non_temporal += time.time() - start_time
            
            matched_obj_idxs = (patch_features_norm @ scans_objs_embeds[target_scan_id].T).argmax(axis=1)
            obj_ids_cpu_scan = data_dict['obj_3D_ids'][target_scan_id]
            matched_obj_obj_ids = obj_ids_cpu_scan[matched_obj_idxs]
            
            # temporal
            room_score_scans_T = {}
            target_scan_id = data_dict['temporal_scan_id_list'][batch_i]
            ## start room retrieval in cpu
            start_time = time.time()
            ## normalize patch features
            scans_to_match = list(set(candidate_scan_ids + [target_scan_id]))
            patch_features_norm = patch_features / np.linalg.norm(patch_features, axis=1, keepdims=True)
            for candidate_scan_id in scans_to_match:
                candidate_obj_embeds = scans_objs_embeds[candidate_scan_id]
                patch_obj_sim = patch_features_norm@candidate_obj_embeds.T
                matched_candidate_obj_sim = np.max(patch_obj_sim, axis=1)
                room_score_scans_T[candidate_scan_id] = matched_candidate_obj_sim.sum()
            room_sorted_by_scores_T = [item[0] for item in sorted(room_score_scans_T.items(), key=lambda x: x[1], reverse=True)]
            for k in top_k_list:
                if target_scan_id in room_sorted_by_scores_T[:k]:
                    top_k_recall_temporal["R@{}_T_S".format(k)] += 1
            retrieval_time_temporal += time.time() - start_time
            
            matched_obj_idxs = (patch_features_norm @ scans_objs_embeds[target_scan_id].T).argmax(axis=1)
            obj_ids_cpu_scan = data_dict['obj_3D_ids'][target_scan_id]
            matched_obj_obj_ids_temp = obj_ids_cpu_scan[matched_obj_idxs]
            
            # retrieva_record
            scan_id = data_dict['scan_ids'][batch_i]
            if scan_id not in room_retrieval_record:
                room_retrieval_record[scan_id] = {'frames_retrieval': {}}
                room_retrieval_record[scan_id]['candidates_scan_ids'] = data_dict['candidate_scan_ids_list'][batch_i]
                room_retrieval_record[scan_id]['obj_ids'] = data_dict['obj_3D_ids'][scan_id]
            frame_idx = data_dict['frame_idxs'][batch_i]
            frame_retrieval = {
                'frame_idx': frame_idx,
                'temporal_scan_id': data_dict['temporal_scan_id_list'][batch_i],
                'matched_obj_obj_ids': matched_obj_obj_ids,
                'matched_obj_obj_ids_temp': matched_obj_obj_ids_temp,
                'room_score_scans_NT': room_score_scans_NT,
                'room_score_scans_T': room_score_scans_T,
            }
            room_retrieval_record[scan_id]['frames_retrieval'][frame_idx] = frame_retrieval

        # average over batch
        for k in top_k_list:
            top_k_recall_temporal["R@{}_T_S".format(k)] /= 1.0*batch_size
            top_k_recall_non_temporal["R@{}_NT_S".format(k)] /= 1.0*batch_size
        retrieval_time_temporal = retrieval_time_temporal / (1.0*batch_size)
        retrieval_time_non_temporal = retrieval_time_non_temporal / (1.0*batch_size)
        
        result = {
            'time_T_S': retrieval_time_temporal,
            'time_NT_S': retrieval_time_non_temporal,
        }
        result.update(top_k_recall_temporal)
        result.update(top_k_recall_non_temporal)
        return result

    def room_retrieval_val(self):
        # val 
        data_dicts = tqdm.tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader))
        for iteration, data_dict in data_dicts:
            result = self.room_retrieval_dict(data_dict, self.val_dataset, self.val_room_retrieval_record, True)
            self.val_room_retrieval_summary.update_from_result_dict(result)
        val_items = self.val_room_retrieval_summary.tostringlist()
        # write metric to file
        val_file = osp.join(self.output_dir, 'val_result.txt')
        common.write_to_txt(val_file, val_items)
        # write retrieval record to file
        retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_val.pkl')
        common.write_pkl_data(self.val_room_retrieval_record, retrieval_record_file)
        
        # test 

        data_dicts = tqdm.tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader))
        for iteration, data_dict in data_dicts:
            result = self.room_retrieval_dict(data_dict, self.test_dataset,self.test_room_retrieval_record, True)
            self.test_room_retrieval_summary.update_from_result_dict(result)

        test_items = self.test_room_retrieval_summary.tostringlist()
        # write metric to file
        test_file = osp.join(self.output_dir, 'test_result.txt')
        common.write_to_txt(test_file, test_items)
        # write retrieval record to file
        retrieval_record_file = osp.join(self.output_dir, 'retrieval_record_test.pkl')
        common.write_pkl_data(self.test_room_retrieval_record, retrieval_record_file)
            

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    
    cfg = update_config_room_retrival(config, args.config, ensure_dir=True)
    
    # copy config file to out dir
    out_dir = osp.join(cfg.output_dir, cfg.val.room_retrieval.method_name)
    common.ensure_dir(out_dir)
    command = 'cp {} {}'.format(args.config, out_dir)
    subprocess.call(command, shell=True)

    tester = RoomRetrivalScore(cfg)
    tester.room_retrieval_val()
    breakpoint = 0

if __name__ == '__main__':
    main()