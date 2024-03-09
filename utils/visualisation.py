from nis import cat
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy
import os.path as osp
import bisect, pickle
from collections import Counter
import sys
sys.path.append('.')
vlsg_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(vlsg_dir)
from utils import common

def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b]).astype(np.float32)

def remove_ceiling(points):
    points_mask = points[..., 2] < np.max(points[..., 2]) - 1
    points = points[points_mask]
    return points

def visualise_dict_counts(counts_dict, title = '', file_name=None):
    class_names = list(counts_dict.keys())
    counts = np.array(list(counts_dict.values()))
    counts = counts.astype(np.float32)
    counts = list(counts)

    fig = plt.figure(figsize = (15, 7.5))
    plt.bar(class_names, counts, color ='#9fb4e3', width = 0.4)
    plt.xticks(rotation=55)
    plt.title(title)
    plt.show()

    if file_name is not None:
        plt.savefig(file_name)

def visualise_point_cloud_registration(src_points, ref_points, gt_transform, est_transform):
    from utils import open3d
    src_point_cloud = open3d.make_open3d_point_cloud(src_points)
    src_point_cloud.estimate_normals()
    src_point_cloud.paint_uniform_color(open3d.get_color("custom_blue"))

    ref_point_cloud = open3d.make_open3d_point_cloud(ref_points)
    ref_point_cloud.estimate_normals()
    ref_point_cloud.paint_uniform_color(open3d.get_color("custom_yellow"))

    open3d.draw_geometries(ref_point_cloud, deepcopy(src_point_cloud).transform(gt_transform))
    open3d.draw_geometries(ref_point_cloud, deepcopy(src_point_cloud).transform(est_transform))

def plotBar(metric_title, x_label, y_label, labels, metric_values, 
            method_names, fig_path, figsize=(12, 9), x_rotation=0):
    # metric_values m x l, m for different methods, l for different semantic classes
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='major', labelsize=14)

    metric_values = np.array(metric_values).reshape(len(method_names), -1)
    num_methods = metric_values.shape[0]
    num_labels = metric_values.shape[1]
    bar_width = min(0.08, 1.0/(num_methods*2) )
    bars = {}

    for m_i in range(num_methods):
        bar_shift = m_i-num_methods/2.0
        bars[m_i] = ax.bar(x + bar_width*bar_shift, metric_values[m_i], bar_width, label=method_names[m_i])

    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_title(metric_title)
    ax.set_xticks(x, rotation=x_rotation)
    ax.set_xticklabels(labels, rotation = x_rotation)
    ax.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches='tight')
    
# get correlation between retrieval score and success rate
class RetrievalStatistics:
    def __init__(self, retrieval_records_dir, retrieval_records = None,
                 temp = True, split = "val"):
        self.undefined = 0
        self.split = split
        self.retrieval_records_dir = retrieval_records_dir
        self.out_dir = osp.join(retrieval_records_dir, "{}_{}_statistics".format(split, "temp" if temp else "static"))
        common.ensure_dir(self.out_dir)
        
        if retrieval_records is not None:
            self.retrieval_records = retrieval_records
        else:
            retrieval_pkl = osp.join(retrieval_records_dir, 
                                    "retrieval_record_{}.pkl".format(split))
            retrieval_records = pickle.load(open(retrieval_pkl, "rb"))
            self.retrieval_records = retrieval_records
            
        self.temp = temp
        self.scan_ids = list(retrieval_records.keys())

    def get_score_and_sucess(self, scan_id, frame_record):
        if self.temp:
            retrieval_scores = frame_record["room_score_scans_T"]
            target_scan_id = frame_record["temporal_scan_id"]
        else:
            retrieval_scores = frame_record["room_score_scans_NT"]
            target_scan_id = scan_id
            
        room_sorted_by_scores = [item[0] for item in sorted(
            retrieval_scores.items(), key=lambda x: x[1], reverse=True)]
        
        success = target_scan_id == room_sorted_by_scores[0]
        score = retrieval_scores[room_sorted_by_scores[0]]
        return score, success
    
    def get_shannonEntropy(self, scan_id, frame_record):
        # get success
        if self.temp:
            gt_obj_cates = frame_record['gt_obj_cates_temp']
        else:
            gt_obj_cates = frame_record['gt_obj_cates']
        
        # get Shannon Entropy
        patch_obj_ids = frame_record['gt_anno']
        patch_obj_ids = patch_obj_ids[patch_obj_ids != self.undefined]
        gt_obj_cates = gt_obj_cates[patch_obj_ids != self.undefined]
        ## merge instances of cateogory
        wall_cate_id = 1
        wall_obj_ids = patch_obj_ids[ gt_obj_cates == wall_cate_id ]
        if len(wall_obj_ids) > 0:
            patch_obj_ids[ gt_obj_cates == wall_cate_id ] = wall_obj_ids[0]
        counts = Counter(patch_obj_ids)
        total = sum(counts.values())
        entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
        return entropy
    
    def get_shannonEntropy_and_sucess(self, scan_id, frame_record):
        # get success
        if self.temp:
            retrieval_scores = frame_record["room_score_scans_T"]
            target_scan_id = frame_record["temporal_scan_id"]
            gt_obj_cates = frame_record['gt_obj_cates_temp']
        else:
            retrieval_scores = frame_record["room_score_scans_NT"]
            target_scan_id = scan_id
            gt_obj_cates = frame_record['gt_obj_cates']
        room_sorted_by_scores = [item[0] for item in sorted(
            retrieval_scores.items(), key=lambda x: x[1], reverse=True)]
        success = target_scan_id == room_sorted_by_scores[0]
        
        # get Shannon Entropy
        entropy = self.get_shannonEntropy(scan_id, frame_record)
        return entropy, success

    def get_shannonEntropy_and_patchsucess(self, scan_id, frame_record):
        # get patch success
        if self.temp:
            patch_predict = frame_record["matched_obj_ids_temp"]
            correct_patch_predict_allscans = frame_record["is_patch_correct_allscans_temp"]
        else:
            patch_predict = frame_record["matched_obj_ids"]
            correct_patch_predict_allscans = frame_record["is_patch_correct_allscans"]
        # patch-match-target scan
        gt_anno = frame_record['gt_anno']
        ## filter out undefined
        patch_predict = patch_predict[gt_anno != self.undefined]
        correct_patch_predict_allscans = correct_patch_predict_allscans[gt_anno != self.undefined]
        gt_anno = gt_anno[gt_anno != self.undefined]
        if len(patch_predict) == 0:
            return None, None, None
        match_success_ratio = np.sum(patch_predict == gt_anno) * 1.0 / len(gt_anno)
        # patch-match-all scans
        match_success_ratio_allscans = np.sum(correct_patch_predict_allscans) * 1.0 / len(correct_patch_predict_allscans)
        
        # get Shannon Entropy
        entropy = self.get_shannonEntropy(scan_id, frame_record)
        return entropy, match_success_ratio, match_success_ratio_allscans
    
    def get_SceneObjNum_and_sucess(self, scan_id, frame_record):
        # get success
        if self.temp:
            retrieval_scores = frame_record["room_score_scans_T"]
            target_scan_id = frame_record["temporal_scan_id"]
        else:
            retrieval_scores = frame_record["room_score_scans_NT"]
            target_scan_id = scan_id
        room_sorted_by_scores = [item[0] for item in sorted(
            retrieval_scores.items(), key=lambda x: x[1], reverse=True)]
        success = target_scan_id == room_sorted_by_scores[0]
        
        # get scene object number
        scene_obj_num = self.retrieval_records[scan_id]['obj_ids'].size
        return scene_obj_num, success

    def get_SceneObjNum_and_patchsucess(self, scan_id, frame_record):
        # get patch success
        if self.temp:
            patch_predict = frame_record["matched_obj_ids_temp"]
            correct_patch_predict_allscans = frame_record["is_patch_correct_allscans_temp"]
        else:
            patch_predict = frame_record["matched_obj_ids"]
            correct_patch_predict_allscans = frame_record["is_patch_correct_allscans"]
        # patch-match-target scan
        gt_anno = frame_record['gt_anno']
        ## filter out undefined
        patch_predict = patch_predict[gt_anno != self.undefined]
        correct_patch_predict_allscans = correct_patch_predict_allscans[gt_anno != self.undefined]
        gt_anno = gt_anno[gt_anno != self.undefined]
        if len(patch_predict) == 0:
            return None, None, None
        match_success_ratio = np.sum(patch_predict == gt_anno) * 1.0 / len(gt_anno)
        # patch-match-all scans
        match_success_ratio_allscans = np.sum(correct_patch_predict_allscans) * 1.0 / len(correct_patch_predict_allscans)
        
        # get scene object number
        scene_obj_num = self.retrieval_records[scan_id]['obj_ids'].size
        return scene_obj_num, match_success_ratio, match_success_ratio_allscans

    def generateScoreAccuCorrelation(self, num_bins = 40, fig_size = (20, 5)):
        scan_ids =  list(self.retrieval_records.keys())
        
        bins = np.linspace(0.4, 1, num_bins + 1)
        num_pos_bins = np.zeros(num_bins)
        num_neg_bins = np.zeros(num_bins)
        retrie_success_ratio = np.zeros(num_bins, dtype=np.float32)
        
        scores_list = []
        success_list = []
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                score, success = self.get_score_and_sucess(scan_id, frame_record)
                patch_num = frame_record['matched_obj_ids'].size
                
                score_normed = score * 1.0 / patch_num
                bin_idx = bisect.bisect(bins, score_normed) -1 
                scores_list.append(score_normed)
                if success:
                    num_pos_bins[bin_idx] += 1
                    success_list.append(1)
                else:
                    num_neg_bins[bin_idx] += 1
                    success_list.append(0)
                    
        retrie_success_ratio = num_pos_bins / (num_pos_bins + num_neg_bins)
        ## normalize to the max value of num_pos_bins and num_neg_bins
        retrie_success_ratio *= max(num_pos_bins.max(), num_neg_bins.max())
        
        # plot
        figure_name = "{}_{}retrieval_score_success.png".format(
            self.split, "temp" if self.temp else "static")
        
        metric_title = "Success with R@1 ~ Retrieval Score"
        x_label = "Score of retried room"
        y_label = "Number of success/fail with R@1"
        labels = ['{:.2f}'.format( (bins[i] + bins[i+1])/2.0 ) for i in range(num_bins)]
        method_names = ['success', 'fail', 'sucess_ratio_retrieval']
        colors = ['g', 'r', 'b']
        metric_values = np.stack([num_pos_bins, num_neg_bins, retrie_success_ratio], axis=0)
        fig_path = osp.join(self.out_dir, figure_name)
        self.plotBar(metric_title, x_label, y_label, labels, metric_values, 
                method_names, fig_path, figsize=fig_size, x_rotation=0, font_size=10, colors=colors)
        
        # calculate correlation
        scores_arr = np.array(scores_list)
        success_arr = np.array(success_list)
        ## calculate correlation
        correlation = np.corrcoef(scores_arr, success_arr)[0, 1]
        return correlation
        
    def generateImgObjAccuCorrelation(self, num_bins = 40, fig_size = (20, 5)):
        scan_ids =  list(self.retrieval_records.keys())
        
        # get shannon entropy and success pairs
        success_list = []
        shannon_entropy_list = []
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            # static case
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                shannon, success = self.get_shannonEntropy_and_sucess(scan_id, frame_record)
                shannon_entropy_list.append(shannon)
                success_list.append(success)
        
        max_entropy = max(shannon_entropy_list) + 1e-6
        min_entropy = min(shannon_entropy_list)
        bins = np.linspace(min_entropy, max_entropy, num_bins+1)
        num_pos_bins = np.zeros(num_bins)
        num_neg_bins = np.zeros(num_bins)
        retrie_success_ratio = np.zeros(num_bins, dtype=np.float32)
        
        for i in range(len(shannon_entropy_list)):
            entropy = shannon_entropy_list[i]
            success = success_list[i]
            bin_idx = bisect.bisect(bins, entropy) -1 
            if success:
                num_pos_bins[bin_idx] += 1
            else:
                num_neg_bins[bin_idx] += 1 
        retrie_success_ratio = num_pos_bins / (num_pos_bins + num_neg_bins)
        ## normalize to the max value of num_pos_bins and num_neg_bins
        retrie_success_ratio *= max(num_pos_bins.max(), num_neg_bins.max())
        
        # plot
        figure_name = "{}_{}_retrie_shanno.png".format(
            self.split, "temp" if self.temp else "static")
        metric_title = "Success with R@1 ~ Shannon Entropy of Objects observed in Images"
        x_label = "Shannon Entropy of Images"
        y_label = "Number of success/fail with R@1"
        labels = ['{:.2f}'.format( (bins[i] + bins[i+1])/2.0 ) for i in range(num_bins)]
        method_names = ['success', 'fail', 'sucess_ratio_retrieval']
        colors = ['g', 'r', 'b']
        metric_values = np.stack([num_pos_bins, num_neg_bins, retrie_success_ratio], axis=0)
        fig_path = osp.join(self.out_dir, figure_name)
        self.plotBar(metric_title, x_label, y_label, labels, metric_values, 
                method_names, fig_path, figsize=fig_size, x_rotation=0, font_size=10, colors=colors)
        
        # calculate correlation
        entropy_arr = np.array(shannon_entropy_list)
        success_arr = np.array(success_list)
        ## calculate correlation
        correlation = np.corrcoef(entropy_arr, success_arr)[0, 1]
        return correlation
  
    def generateImgObjPatchAccuCorrelation(self, num_bins = 40, fig_size = (20, 5)):
        scan_ids =  list(self.retrieval_records.keys())
        
        # get shannon entropy and success pairs
        success_ratio_list = []
        success_ratio_all_scans_list = []
        shannon_entropy_list = []
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                shannon, success, success_allscans = \
                    self.get_shannonEntropy_and_patchsucess(scan_id, frame_record)
                # skip if no patch
                if shannon is None or success is None or success_allscans is None:
                    continue
                
                shannon_entropy_list.append(shannon)
                success_ratio_list.append(success)
                success_ratio_all_scans_list.append(success_allscans)
        
        max_entropy = max(shannon_entropy_list) + 1e-6
        min_entropy = min(shannon_entropy_list)
        bins = np.linspace(min_entropy, max_entropy, num_bins+1)
        success_ratio = [[] for _ in range(num_bins)]
        
        for i in range(len(shannon_entropy_list)):
            entropy = shannon_entropy_list[i]
            bin_idx = bisect.bisect(bins, entropy) -1 
            success_ratio[bin_idx].append(success_ratio_list[i])
            
        invalid_bins = [0 for _ in range(num_bins)]
        for b_i in range(num_bins):
            if len(success_ratio[b_i]) == 0:
                success_ratio[b_i] = 0
                invalid_bins[bin_idx] = 1
            else:
                success_ratio[b_i] = np.mean(np.array(success_ratio[b_i]))
        success_ratio = np.array(success_ratio)
        invalid_bins = np.array(invalid_bins)
        # remove invalid success_ratio
        success_ratio = success_ratio[invalid_bins == 0]
        
        # plot
        figure_name = "{}_{}_patch_shanno.png".format(
            self.split, "temp" if self.temp else "static")
        metric_title = "Patch-Object Success Ratio ~ Shannon Entropy of Objects observed in Images"
        x_label = "Shannon Entropy of Images"
        y_label = "Patch-Object Success Ratio"
        labels = ['{:.2f}'.format( (bins[i] + bins[i+1])/2.0 ) for i in range(num_bins)]
        labels = [l for i, l in enumerate(labels) if invalid_bins[i] == 0]
        method_names = ['patch_obj_match_success_ratio']
        colors = ['g']
        metric_values = np.stack([success_ratio], axis=0)
        fig_path = osp.join(self.out_dir, figure_name)
        self.plotBar(metric_title, x_label, y_label, labels, metric_values, 
                method_names, fig_path, figsize=fig_size, x_rotation=0, font_size=10, colors=colors)
        
        # calculate correlation
        entropy_arr = np.array(shannon_entropy_list)
        success_arr = np.array(success_ratio_list)
        success_arr_allscans = np.array(success_ratio_all_scans_list)
        ## calculate correlation
        correlation_entropy_success = np.corrcoef(entropy_arr, success_arr)[0, 1]
        correlation_entropy_success_allscans = np.corrcoef(
            entropy_arr, success_arr_allscans)[0, 1]
        
        return correlation_entropy_success, correlation_entropy_success_allscans, np.mean(success_arr), np.mean(success_arr_allscans)
       
    def generateSceneObjAccuCorrelation(self, num_bins = 40, fig_size = (20, 5)):
        scan_ids =  list(self.retrieval_records.keys())
        
        # get shannon entropy and success pairs
        success_ratio_list = []
        scene_obj_num_list = []
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            # static case
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                scene_obj_num, success = self.get_SceneObjNum_and_sucess(scan_id, frame_record)
                scene_obj_num_list.append(scene_obj_num)
                success_ratio_list.append(success)
        
        # calculate correlation
        scene_obj_num_arr = np.array(scene_obj_num_list)
        success_arr = np.array(success_ratio_list)
        ## calculate correlation
        correlation = np.corrcoef(scene_obj_num_arr, success_arr)[0, 1]
        return correlation
    
    def generateSceneObjPatchAccuCorrelation(self, num_bins = 40, fig_size = (20, 5)):
        scan_ids =  list(self.retrieval_records.keys())
        
        # get scene obj num and success pairs
        success_ratio_list = []
        success_ratio_all_scans_list = []
        scene_obj_num_list = []
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                scene_obj_num, success, success_allscans = \
                    self.get_SceneObjNum_and_patchsucess(scan_id, frame_record)
                # skip if no patch
                if scene_obj_num is None or success is None or success_allscans is None:
                    continue
                
                scene_obj_num_list.append(scene_obj_num)
                success_ratio_list.append(success)
                success_ratio_all_scans_list.append(success_allscans)
        
        # calculate correlation
        scene_obj_num_arr = np.array(scene_obj_num_list)
        success_arr = np.array(success_ratio_list)
        success_arr_allscans = np.array(success_ratio_all_scans_list)
        ## calculate correlation
        correlation_obj_num_success = np.corrcoef(scene_obj_num_arr, success_arr)[0, 1]
        correlation_obj_num_success_allscans = np.corrcoef(
            scene_obj_num_arr, success_arr_allscans)[0, 1]
        
        return correlation_obj_num_success, correlation_obj_num_success_allscans
       
    def generateSemanticConfusionMatrix(self, topk = 20):
        scan_ids =  list(self.retrieval_records.keys())
        sem_cat_id2name = self.retrieval_records[scan_ids[0]]['sem_cat_id2name']
        sem_num = len(sem_cat_id2name)
        confusion_matrix = np.zeros((sem_num, sem_num))
        confusion_matrix_allscans = np.zeros((sem_num, sem_num))
        for scan_id in scan_ids:
            record = self.retrieval_records[scan_id]
            frames_retrieval = record["frames_retrieval"]
            frame_idxs = list(frames_retrieval.keys())
            
            for frame_idx in frame_idxs:
                frame_record = frames_retrieval[frame_idx]
                if self.temp:
                    matched_obj_cates = frame_record["matched_obj_cates_temp"]
                    matched_obj_cates_allscans = frame_record["matched_obj_cates_allscans_temp"]
                    gt_obj_cates = frame_record["gt_obj_cates_temp"]
                else:
                    matched_obj_cates = frame_record["matched_obj_cates"]
                    matched_obj_cates_allscans = frame_record["matched_obj_cates_allscans"]
                    gt_obj_cates = frame_record["gt_obj_cates"]
                
                # update confusion matrix
                for patch_i in range(gt_obj_cates.size):
                    gt_cate = gt_obj_cates[patch_i]
                    if gt_cate == self.undefined:
                        continue
                    matched_cate = matched_obj_cates[patch_i]
                    matched_cate_allscans = matched_obj_cates_allscans[patch_i]
                    confusion_matrix[gt_cate-1, matched_cate-1] += 1
                    confusion_matrix_allscans[gt_cate-1, matched_cate_allscans-1] += 1
                
        # select topk semantic categories
        cate_num = np.sum(confusion_matrix, axis=1)
        cates_topk_idxs = np.argsort(cate_num)[::-1][:topk]
        ## re-order cates_topk_idxs with origianl cate idx
        cates_topk_idxs = np.sort(cates_topk_idxs)
        
        confusion_matrix = confusion_matrix[cates_topk_idxs][:, cates_topk_idxs]
        confusion_matrix_allscans = confusion_matrix_allscans[cates_topk_idxs][:, cates_topk_idxs]
        ## update cate nums
        sem_num = topk
        sem_cat_id2name = {i+1: sem_cat_id2name[i+1] for i in cates_topk_idxs}
        sem_cat_idxs = [i+1 for i in cates_topk_idxs]
        
        # save confusion matrix
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix_allscans = confusion_matrix_allscans / np.sum(confusion_matrix_allscans, axis=1, keepdims=True)
        confusion_matrix_file = osp.join(self.out_dir, "{}_{}_confusion_matrix.pkl".format(self.split, "temp" if self.temp else "static"))
        confusion_matrix_allscans_file = osp.join(self.out_dir, "{}_{}_confusion_matrix_allscans.pkl".format(self.split, "temp" if self.temp else "static"))
        np.save(confusion_matrix_file, confusion_matrix)
        np.save(confusion_matrix_allscans_file, confusion_matrix_allscans)
        
        # generate confusion matrix figure
        ## color map
        colors = [(0.5, 0, 0), (0.5, 0.2, 0), (0.5, 0.5, 0), (0, 0.5, 0), (0, 0.2, 0.5), (0, 0, 0.5)] #  purple to red
        ## reverse
        colors = colors[::-1]
        n_bins = 100  # Increase this number for a smoother transition
        cmap_name = 'cate_confusion_matrix'
        cmap_confusion_matrix = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        
        figure_name = "{}_{}_confusion_matrix.png".format(
            self.split, "temp" if self.temp else "static")
        fig_path = osp.join(self.out_dir, figure_name)
        
        fig_size = (23, 23)
        label_font_size = 28
        cate_name_font_size = 22
        plt.figure(figsize=fig_size)
        img = plt.imshow(confusion_matrix, cmap=cmap_confusion_matrix, interpolation='nearest')  # Store the AxesImage object
        plt.xticks(range(sem_num), [sem_cat_id2name[i] for i in sem_cat_idxs]
                   , rotation=45, fontsize=cate_name_font_size)
        plt.yticks(range(sem_num), [sem_cat_id2name[i] for i in sem_cat_idxs],
                   fontsize=cate_name_font_size)
        plt.xlabel('Predicted Semantic Category', fontsize=label_font_size)
        plt.ylabel('Ground Truth Semantic Category', fontsize=label_font_size)
        cbar_ax = plt.gcf().add_axes([0.25, 0.95, 0.5, 0.03])  # Adjust these values as needed
        plt.colorbar(img, cax=cbar_ax, orientation='horizontal')  # Set the colorbar in the new axes
        plt.savefig(fig_path)
        
        figure_name_allscans = "{}_{}_confusion_matrix_allscans.png".format(
            self.split, "temp" if self.temp else "static")
        fig_path_allscans = osp.join(self.out_dir, figure_name_allscans)
        fig_path = osp.join(self.out_dir, figure_name_allscans)
        plt.figure(figsize=fig_size)
        img = plt.imshow(confusion_matrix_allscans, cmap=cmap_confusion_matrix, interpolation='nearest')  # Store the AxesImage object
        plt.xticks(range(sem_num), [sem_cat_id2name[i] for i in sem_cat_idxs]
                   , rotation=45, fontsize=cate_name_font_size)
        plt.yticks(range(sem_num), [sem_cat_id2name[i] for i in sem_cat_idxs],
                   fontsize=cate_name_font_size)
        plt.xlabel('Predicted Semantic Category', fontsize=label_font_size)
        plt.ylabel('Ground Truth Semantic Category', fontsize=label_font_size)
        cbar_ax = plt.gcf().add_axes([0.25, 0.95, 0.5, 0.03])  # Adjust these values as needed
        plt.colorbar(img, cax=cbar_ax, orientation='horizontal')  # Set the colorbar in the new axes
        plt.savefig(fig_path_allscans)     
        return confusion_matrix

    def generateStaistics(self):
        ScoreAccuCorre = self.generateScoreAccuCorrelation()
        ImgObjAccuCorre = self.generateImgObjAccuCorrelation()
        SceneObjAccuCorre = self.generateSceneObjAccuCorrelation()
        
        ImgObjPatchAccuCorre, ImgObjPatchAllscansAccuCorre, PatchAccu, PatchAccuAllScans = \
            self.generateImgObjPatchAccuCorrelation()
        SceneObjPatchAccuCorre, SceneObjPatchAllscansAccuCorre = self.generateSceneObjPatchAccuCorrelation()
        self.generateSemanticConfusionMatrix()

        # save to txt
        txt_file = osp.join(self.out_dir, "{}_retrieval_statistics.txt".format(self.split))
        with open(txt_file, "w") as f:
            f.write("Score~R1 Pearson Correlation Coeff: {}\n".format(ScoreAccuCorre))
            f.write("ImgObjShannon~R1 Pearson Correlation Coeff: {}\n".format(ImgObjAccuCorre))
            f.write("SceneObj~R1 Pearson Correlation Coeff: {}\n".format(SceneObjAccuCorre))
            
            f.write("PatchR1: {}\n".format(PatchAccu))
            f.write("ImgObjShannon~PatchR1 Pearson Correlation Coeff: {}\n".format(ImgObjPatchAccuCorre))
            f.write("SceneObj~PatchR1 Pearson Correlation Coeff: {}\n".format(SceneObjPatchAccuCorre))
            
            f.write("PatchAccu (all scans as candidates): {}\n".format(PatchAccuAllScans))
            f.write("ImgObjShannon~PatchR1 Pearson Correlation Coeff (all scans as candidates): {}\n".format(ImgObjPatchAllscansAccuCorre))
            f.write("SceneObj~PatchR1 Pearson Correlation Coeff (all scans as candidates): {}\n".format(SceneObjPatchAllscansAccuCorre))
            
    def plotBar(self, metric_title, x_label, y_label, labels, metric_values, 
                method_names, fig_path, figsize=(12, 9), x_rotation=0, 
                font_size=14, colors=[]):
        # metric_values m x l, m for different methods, l for different semantic classes
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=figsize)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        metric_values = np.array(metric_values).reshape(len(method_names), -1)
        num_methods = metric_values.shape[0]
        num_labels = metric_values.shape[1]
        bar_width = min(0.15, 1.0/(num_methods*1.5) )
        bars = {}

        for m_i in range(num_methods):
            bar_shift = m_i-num_methods/2.0
            bars[m_i] = ax.bar(x + bar_width*bar_shift, metric_values[m_i], 
                               bar_width, label=method_names[m_i], color=colors[m_i])

        ax.set_ylabel(y_label, fontsize=font_size)
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_title(metric_title)
        ax.set_xticks(x, rotation=x_rotation)
        ax.set_xticklabels(labels, rotation = x_rotation)
        ax.legend(loc='upper left', fontsize=font_size)
        fig.tight_layout()
        fig.savefig(fig_path, bbox_inches='tight')
        