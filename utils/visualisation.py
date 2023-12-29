import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
sys.path.append('.')
from utils import open3d

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