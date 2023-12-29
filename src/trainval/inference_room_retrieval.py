import argparse
import os 
import os.path as osp
import time
import comm
from matplotlib import patches
import numpy as np 
import sys
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ws_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(ws_dir)

# utils
from utils import common
from utils import visualisation

# config
from configs import update_config_room_retrival, config
# tester
from engine.single_tester import SingleTester
from utils import torch_util
from datasets.loaders import get_train_val_data_loader, get_val_dataloader
from datasets.scan3r_obj_pair_cross_scenes import PatchObjectPairCrossScenesDataSet
# models
import torch
import torch.optim as optim
from mmdet.models import build_backbone
from mmcv import Config
# from models.GCVit.models import gc_vit
from models.patch_obj_aligner import PatchObjectAligner
from models.loss import ICLLoss
from models.path_obj_pair_visualizer import PatchObjectPairVisualizer

# use PathObjAligner for room retrieval
class RoomRetrivalPatchObjAligner(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=None)
        
        # cfg
        self.cfg = cfg 
        self.method_name = cfg.val.method_name
        
        # files
        self.output_dir = osp.join(cfg.output_dir, self.method_name)
        common.ensure_dir(self.output_dir)
        
        # sgaliner related cfg
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
        self.scan_type = cfg.sgaligner.scan_type
        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.cate_file = osp.join(self.scans_files_dir, 'scannet40_classes.txt')
        # get cate info
        self.cate_info = common.idx2name(self.cate_file)
        
        # dataloader
        start_time = time.time()
        dataset, data_loader = get_val_dataloader(cfg, Dataset = PatchObjectPairCrossScenesDataSet)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        self.register_loader(data_loader)
        self.register_dataset(dataset)
        
        # model
        self.registerPatchObjectAlignerFromCfg(cfg)
        self.model.eval()
        
        # result
        self.patches_result_overall = {'correct': 0, 'total': 0} # overall patch match result
        self.patches_result_nyu = {} # patch match result per category (nyu)
        self.patches_result_scan_id = {} # patch match result per scan
        self.patches_accuracy_image = [] # image-level patch match accuracy
        
        self.room_retrieval_result = {'correct': 0, 'total': 0, 'correct_score': [], 
                                      'correct_match': []} # overall room retrieval result
        self.room_retrieval_scan_id = {} # room retrieval result per scan
        
    def registerPatchObjectAlignerFromCfg(self, cfg):
        # load backbone
        backbone_cfg_file = cfg.model.backbone.cfg_file
        # ugly hack to load pretrained model, maybe there is a better way
        backbone_cfg = Config.fromfile(backbone_cfg_file)
        backbone_pretrained_file = cfg.model.backbone.pretrained
        backbone_cfg.model['backbone']['pretrained'] = backbone_pretrained_file
        backbone = build_backbone(backbone_cfg.model['backbone'])
        
        # get patch object aligner
        num_reduce = cfg.model.backbone.num_reduce
        backbone_dim = cfg.model.backbone.backbone_dim
        img_rotate = cfg.data.img_encoding.img_rotate
        
        patch_hidden_dims = cfg.model.patch.hidden_dims
        patch_encoder_dim = cfg.model.patch.encoder_dim
        
        obj_embedding_dim = cfg.model.obj.embedding_dim
        obj_embedding_hidden_dims = cfg.model.obj.embedding_hidden_dims
        obj_encoder_dim = cfg.model.obj.encoder_dim
        
        drop = cfg.model.other.drop
        
        self.model = PatchObjectAligner(backbone,
                                num_reduce,
                                backbone_dim,
                                img_rotate, 
                                patch_hidden_dims,
                                patch_encoder_dim,
                                obj_embedding_dim,
                                obj_embedding_hidden_dims,
                                obj_encoder_dim,
                                drop)
        
        # load snapshot
        if (cfg.other.use_resume and os.path.isfile(cfg.other.resume)):
            self.snap_shot = cfg.other.resume
            self.load_snapshot(cfg.other.resume)
        else:
            raise RuntimeError('No snapshot is provided.')
        
        # model to cuda 
        self.model.to(self.device)
        
        # log
        message = 'Model description:\n' + str(self.model)
        self.logger.info(message) 

    def recordPatchMatchResult(self, patch_info):
        # record overall result
        if patch_info['is_obj_correct']:
            self.patches_result_overall['correct'] += 1
        self.patches_result_overall['total'] += 1
        
        # record result per scan
        scan_id = patch_info['scan_id']
        if scan_id not in self.patches_result_scan_id:
            self.patches_result_scan_id[scan_id] = {'correct': 0, 'total': 0}
        if patch_info['is_obj_correct']:
            self.patches_result_scan_id[scan_id]['correct'] += 1
        self.patches_result_scan_id[scan_id]['total'] += 1
        
        # record result per category
        obj_cate = patch_info['gt_obj_cate']
        if obj_cate not in self.patches_result_nyu:
            self.patches_result_nyu[obj_cate] = {'correct': 0, 'total': 0}
        if patch_info['is_obj_correct']:
            self.patches_result_nyu[obj_cate]['correct'] += 1
        self.patches_result_nyu[obj_cate]['total'] += 1
        
    def recordImagePatchMatchAccuracy(self, accuracy):
        self.patches_accuracy_image.append(accuracy)
        
    def recordRoomRetrievalResult(self, retrieval_info):
        # record overall result
        if retrieval_info['is_room_correct']:
            self.room_retrieval_result['correct'] += 1
            self.room_retrieval_result['correct_score'].append(
                retrieval_info['selected_scan_id_votes'] * 1.0/retrieval_info['total_votes'])
            self.room_retrieval_result['correct_match'].append(
                retrieval_info['num_votes_correct_match'] * 1.0/retrieval_info['total_votes'])
        self.room_retrieval_result['total'] += 1
        
        # record result per scan
        scan_id = retrieval_info['scan_id']
        if scan_id not in self.room_retrieval_scan_id:
            self.room_retrieval_scan_id[scan_id] = {'correct': 0, 'total': 0, 'correct_score': [],
                                                    'correct_match': []}
        if retrieval_info['is_room_correct']:
            self.room_retrieval_scan_id[scan_id]['correct'] += 1
            self.room_retrieval_scan_id[scan_id]['correct_score'].append(
                retrieval_info['selected_scan_id_votes'] * 1.0/retrieval_info['total_votes'])
            self.room_retrieval_scan_id[scan_id]['correct_match'].append(
                retrieval_info['num_votes_correct_match'] * 1.0/retrieval_info['total_votes'])
        self.room_retrieval_scan_id[scan_id]['total'] += 1

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict
    
    def eval_step(self, iteration, data_dict, output_dict):
        # # visualise
        # if self.cfg.test.vis:
        #     self.visualiser.visualise(data_dict, output_dict)
        # return output_dict
        
        # get results
        embs = output_dict
        patch_obj_sim_list = embs['patch_obj_sim']
        batch_size = data_dict['batch_size']
        
        matched_success_batch = None
        for batch_i in range(batch_size):
            # dataitem info
            scan_id = data_dict['scan_ids'][batch_i]
            frame_idx = data_dict['frame_idxs'][batch_i]
            e1i_matrix = data_dict['e1i_matrix_list'][batch_i]
            ## mask out unmatched patches
            e1i_valid = e1i_matrix.sum(dim=-1) > 0 # (N_P)
            ## obj info
            obj_3D_idx2info = data_dict['obj_3D_idx2info_list'][batch_i]
            obj_3D_id2idx = data_dict['obj_3D_id2idx'][batch_i]
            ## gt annotation of obj idx
            obj_2D_patch_anno_flatten = \
                data_dict['obj_2D_patch_anno_flatten_list'][batch_i]
                
            # patch_obj_sim
            patch_obj_sim = patch_obj_sim_list[batch_i]
            matched_obj_idxs = torch.argmax(patch_obj_sim, dim=1).reshape(-1,1) # (N_P)
            
            vote_scans = {} # vote for scan_id given matched obj
            num_votes_correct_match = 0
            for patch_i in range(len(e1i_valid)):

                # get matched obj info
                matched_obj_idx = matched_obj_idxs[patch_i].item()
                matched_obj_info = obj_3D_idx2info[matched_obj_idx]

                # record patch match result
                if e1i_valid[patch_i]:
                    # get gt obj info
                    gt_obj_id = obj_2D_patch_anno_flatten[patch_i].item()
                    gt_obj_idx = obj_3D_id2idx[gt_obj_id]
                    gt_obj_info = obj_3D_idx2info[gt_obj_idx]
                    # get patch matching info
                    patch_info = {
                        'scan_id': scan_id,
                        'frame_idx': frame_idx,
                        'patch_idx': patch_i,
                        'gt_obj_cate': gt_obj_info[2],
                        'is_obj_correct': e1i_matrix[patch_i][matched_obj_idx],
                        'matched_scan_id': matched_obj_info[0],
                    }
                    # aggregate patch match result
                    self.recordPatchMatchResult(patch_info)
                    num_votes_correct_match += e1i_matrix[patch_i][matched_obj_idx]
                
                # counting for room retrieval
                if matched_obj_info[0] not in vote_scans:
                    vote_scans[matched_obj_info[0]] = 1
                else:
                    vote_scans[matched_obj_info[0]] += 1
            
            # get image-level patch match accuracy
            matched_obj_labels = e1i_matrix.gather(1, matched_obj_idxs)
            matched_obj_labels = matched_obj_labels.squeeze(1) # (N_P)
            matched_obj_labels = matched_obj_labels[e1i_valid]
            matched_success_batch = matched_obj_labels if matched_success_batch is None \
                else torch.cat((matched_success_batch, matched_obj_labels), dim=0)
            
            # get room retrieval result
            selected_scan_id = max(vote_scans, key=vote_scans.get)
            selected_scan_id_votes = vote_scans[selected_scan_id]
            retrieval_info = {
                'scan_id': scan_id,
                'selected_scan_id': selected_scan_id,
                'is_room_correct': scan_id == selected_scan_id,
                'selected_scan_id_votes': selected_scan_id_votes,
                'num_votes_correct_match': num_votes_correct_match,
                'total_votes': len(e1i_valid),
            }
            # record room retrieval result
            self.recordRoomRetrievalResult(retrieval_info)
            breakpoint = 0

        self.recordImagePatchMatchAccuracy(matched_success_batch.float().mean())
            
    def after_test_epoch(self):
        # save patch result
        meta_pkl_dir = osp.join(self.output_dir, 'meta')
        common.ensure_dir(meta_pkl_dir)
        patches_result_overall_pkl = osp.join(meta_pkl_dir, 'patches_result_overall.pkl')
        patches_result_nyu_pkl = osp.join(meta_pkl_dir, 'patches_result_nyu.pkl')
        patches_result_scans_pkl = osp.join(meta_pkl_dir, 'patches_result_scans.pkl')
        common.write_pkl_data(self.patches_result_overall, patches_result_overall_pkl)
        common.write_pkl_data(self.patches_result_nyu, patches_result_nyu_pkl)
        common.write_pkl_data(self.patches_result_scan_id, patches_result_scans_pkl)
        
        patches_result_out_file = osp.join(self.output_dir, 'patches_result.txt')
        patches_result_out_scans_file = osp.join(self.output_dir, 'patches_result_scans.txt')
        patches_result_fig_file = osp.join(self.output_dir, 'patches_result_cate.png')
        
        # save retrieval result
        room_retrieval_result_pkl = osp.join(meta_pkl_dir, 'room_retrieval_result.pkl')
        room_retrieval_result_scans_pkl = osp.join(meta_pkl_dir, 'room_retrieval_result_scans.pkl')
        common.write_pkl_data(self.room_retrieval_result, room_retrieval_result_pkl)
        common.write_pkl_data(self.room_retrieval_scan_id, room_retrieval_result_scans_pkl)
        
        room_retrieval_out_scans_file = osp.join(self.output_dir, 'room_retrieval_accu_scans.txt')
        
        # save and visualize patch result
        patches_result_cate_out_lines = []
        patches_result_cate_out_dict = {}
        patches_result_scans_out_lines = []
        ## record overall accuracy
        key = 'overall'
        patches_result_overall = [self.patches_result_overall['correct']*1.0/self.patches_result_overall['total'],
                                   self.patches_result_overall['correct'],self.patches_result_overall['total'] ]
        patches_result_cate_out_lines.append("{}: {:.3f} ({}/{})".format(key, *patches_result_overall))
        patches_result_scans_out_lines.append("{}: {:.3f} ({}/{})".format(key, *patches_result_overall))
        ## record image-level accuracy
        patches_result_overall_image = common.ave_list(self.patches_accuracy_image)
        key = 'overall_per_image'
        patches_result_scans_out_lines.append("{}: {:.3f}".format(key, patches_result_overall_image))
        ## records per category
        keys_cate_sort = list(self.patches_result_nyu.keys())
        keys_cate_sort.sort()
        for key in keys_cate_sort:
            patches_result = [self.patches_result_nyu[key]['correct']*1.0/self.patches_result_nyu[key]['total'],
                                self.patches_result_nyu[key]['correct'], self.patches_result_nyu[key]['total']]
            cate_name = self.cate_info[key]
            patches_result_cate_out_lines.append("{} - {}: {:.3f} ({}/{})".format(key, cate_name, *patches_result))
            patches_result_cate_out_dict[key] = patches_result[0]
        ## records per scan
        key_scan_sort = list(self.patches_result_scan_id.keys())
        key_scan_sort.sort()
        for key in key_scan_sort:
            patches_result = [self.patches_result_scan_id[key]['correct']*1.0/self.patches_result_scan_id[key]['total'],
                                self.patches_result_scan_id[key]['correct'], self.patches_result_scan_id[key]['total']]
            patches_result_scans_out_lines.append("{}: {:.3f} ({}/{})".format(key, *patches_result))
        ## write to file
        with open(patches_result_out_file, 'w') as f:
            for line in patches_result_cate_out_lines:
                f.write(line + '\n')
        
        with open(patches_result_out_scans_file, 'w') as f:
            for line in patches_result_scans_out_lines:
                f.write(line + '\n')
        ## draw figure for patches result per category
        x_label = 'category'
        y_label = 'patch match accuracy'
        title = 'Patch match accuracy per category'
        labels = [self.cate_info[key] for key in keys_cate_sort] + ['overall']
        metric_values = [patches_result_cate_out_dict[key] for key in keys_cate_sort] + [patches_result_overall[0]]
        for key in patches_result_cate_out_dict.keys():
            patches_result_cate_out_dict[key] = patches_result_cate_out_dict[key] * 100
        
        ## draw and save figure
        visualisation.plotBar(metric_title=title, x_label=x_label, y_label=y_label, labels=labels, metric_values=[metric_values], 
                              method_names=[self.method_name], fig_path = patches_result_fig_file, x_rotation=90)
        
        
        # save room retrieval result
        room_retrieval_scans_out_lines = []
        ## record overall accuracy
        key = 'overall'
        room_retrieval_result_overall = [self.room_retrieval_result['correct']*1.0/self.room_retrieval_result['total'],
                                    self.room_retrieval_result['correct'], self.room_retrieval_result['total'], 
                                    common.ave_list( self.room_retrieval_result['correct_score']) , 
                                    common.ave_list(self.room_retrieval_result['correct_match']) ]
        room_retrieval_scans_out_lines.append("{}: {:.3f} ({}/{}), score: {:.3f}, correct_match: {:.3f}".format(key, *room_retrieval_result_overall))
        ## records per scan
        key_scan_sort = list(self.room_retrieval_scan_id.keys())
        key_scan_sort.sort()
        for key in key_scan_sort:
            room_retrieval_result = [self.room_retrieval_scan_id[key]['correct']*1.0/self.room_retrieval_scan_id[key]['total'],
                                self.room_retrieval_scan_id[key]['correct'], self.room_retrieval_scan_id[key]['total'],
                                common.ave_list(self.room_retrieval_scan_id[key]['correct_score']), 
                                common.ave_list(self.room_retrieval_scan_id[key]['correct_match']) ]
            room_retrieval_scans_out_lines.append("{}: {:.3f} ({}/{}), score: {:.3f}, correct_match: {:.3f}".format(key, *room_retrieval_result))
        ## write to file
        with open(room_retrieval_out_scans_file, 'w') as f:
            for line in room_retrieval_scans_out_lines:
                f.write(line + '\n')
                
        breakpoint = 0
        
def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    
    cfg = update_config_room_retrival(config, args.config, ensure_dir=True)
    
    # copy config file to out dir
    out_dir = osp.join(cfg.output_dir, cfg.val.method_name)
    common.ensure_dir(out_dir)
    command = 'cp {} {}'.format(args.config, out_dir)
    subprocess.call(command, shell=True)

    tester = RoomRetrivalPatchObjAligner(cfg)
    tester.run()
    breakpoint = 0

if __name__ == '__main__':
    main()