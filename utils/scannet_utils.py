import os.path as osp
import numpy as np
import json
from glob import glob
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull
from copy import deepcopy
import cv2
import pickle
from utils import common, scan3r, point_cloud, define
import os

def load_frame_idxs(data_split_dir, scan_id, skip=None):
    num_frames = len(glob(osp.join(data_split_dir, scan_id, 'color', '*.jpg')))

    if skip is None:
        frame_idxs = ['{:d}'.format(frame_idx) for frame_idx in range(0, num_frames)]
    else:
        frame_idxs = ['{:d}'.format(frame_idx) for frame_idx in range(0, num_frames, skip)]
    return frame_idxs

def load_frame_paths(data_split_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(data_split_dir, scan_id, skip)
    img_folder = osp.join(data_split_dir, scan_id, 'color')
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = "{}.jpg".format(frame_idx)
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths

def scenegraphfusion2scan3r(scan_id, prediction_folder, edge2idx, class2idx, cfg):
    
    inseg_ply_file = osp.join(prediction_folder, 'inseg.ply')
    pred_file = osp.join(prediction_folder, 'predictions.json')
    ply_save_file = osp.join(prediction_folder, 'inseg_filtered.ply')
    cloud_pd, points_pd, segments_pd = point_cloud.load_inseg(inseg_ply_file)
    filter_seg_size = cfg.preprocess.filter_segment_size
    # get num of segments
    segment_ids = np.unique(segments_pd) 
    segment_ids = segment_ids[segment_ids!=0]
    segments_pd_filtered=list()
    for seg_id in segment_ids:
        pts = points_pd[np.where(segments_pd==seg_id)]
        if len(pts) > filter_seg_size: segments_pd_filtered.append(seg_id)
    segment_ids = segments_pd_filtered
    sgfusion_pred = common.load_json(pred_file)[scan_id]
    rel_obj_data_dict = get_pred_obj_rel(sgfusion_pred, edge2idx, class2idx)
    
    # fuse pred file info and inseg.ply info
    relationships = []
    objects = []
    filtered_segments_ids = []
    ## get segments in both inseg.ply and pred file
    for object_data in rel_obj_data_dict['objects']:
        if int(object_data['id']) in segment_ids: filtered_segments_ids.append(int(object_data['id']))
    segment_ids = filtered_segments_ids
    ## get relationships in both inseg.ply and pred file
    for rel in rel_obj_data_dict['relationships']:
        if int(rel[0]) in segment_ids and int(rel[1]) in segment_ids:
            relationships.append(rel)
    ## get objects in both inseg.ply and pred file
    for seg_id in segment_ids:
        obj_data = [object_data for object_data in rel_obj_data_dict['objects'] if seg_id == int(object_data['id'])]
        if len(obj_data) == 0: continue
        objects.append(obj_data[0])
    assert len(segment_ids) == len([object_data['id'] for object_data in objects])
    ## get points
    points_pd_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(points_pd_mask == True)[0]]
    segments_pd = segments_pd[np.where(points_pd_mask == True)[0]]
    ## filter and merge same part point cloud
    segment_ids, segments_pd, objects, relationships = filter_merge_same_part_pc(segments_pd, objects, relationships)
    assert len(segment_ids) == len([object_data['id'] for object_data in objects])
    ## get relationship and object data
    relationship_data_dict = {'relationships' : relationships}
    object_data_fict = {'objects' : objects}
    ## create inseg_filter ply file
    segments_ids_pc_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(segments_ids_pc_mask == True)[0]]
    segments_pd = segments_pd[np.where(segments_ids_pc_mask == True)[0]]
    verts = []
    for idx, v in enumerate(points_pd):
        vert = (v[0], v[1], v[2], segments_pd[idx])
        verts.append(vert)
    
    verts = np.asarray(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u2')])
    plydata = PlyData([PlyElement.describe(verts, 'vertex', comments=['vertices'])])
    with open(ply_save_file, mode='wb') as f: PlyData(plydata).write(f)
    
    # create scan3r data
    x = verts['x']
    y = verts['y']
    z = verts['z']
    object_id = verts['label']
    vertices = np.empty(x.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('objectId', 'h')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['objectId'] = object_id.astype('h')
    ply_data_pred = vertices
    data_dict = process_ply_data2scan3r(scan_id, ply_data_pred, 
                    relationship_data_dict, object_data_fict, edge2idx, cfg)
    return data_dict
    
def get_pred_obj_rel(sgfusion_pred, edge2idx, class2idx):
    idx2egde = {idx: edge for edge, idx in edge2idx.items()}
    idx2class = {idx: classname for classname, idx in class2idx.items()}
    preds = sgfusion_pred

    # Edges 
    relationships = []
    pred_edges    = preds['edges']
    
    for edge in pred_edges.keys():
        sub = edge.split('_')[0]
        obj = edge.split('_')[1]

        edge_log_softmax = list(pred_edges[edge].values())
        edge_probs = common.log_softmax_to_probabilities(edge_log_softmax)
        edge_id = np.argmax(edge_probs)
        edge_name = idx2egde[edge_id]
        if edge_name not in ['none'] and edge_name is not None:
            relationships.append([str(sub), str(obj), str(edge_id), edge_name])
    
    # Objects
    objects = []
    object_data = preds['nodes']
    
    for object_id in object_data:
        obj_log_softmax =  list(object_data[object_id].values())
        obj_probs = common.log_softmax_to_probabilities(obj_log_softmax)

        obj_id = np.argmax(obj_probs)
        obj_name = idx2class[obj_id]
        
        if obj_name not in ['none'] and obj_name is not None: 
            objects.append({'label' : obj_name, 'id' : str(object_id), 
                            'global_id' : str(obj_id)})
    
    return {'relationships' : relationships, 'objects' : objects}
def filter_merge_same_part_pc(segments_pd, objects, relationships):
    instanceid2objname = {int(object_data['id']) : object_data['label'] for object_data in objects}

    pairs = []
    filtered_relationships = []
    for relationship in relationships:
        if relationship[-1] != define.NAME_SAME_PART : 
            filtered_relationships.append(relationship)
        elif relationship[-1] == define.NAME_SAME_PART and instanceid2objname[int(relationship[0])] == instanceid2objname[int(relationship[1])]:
            pairs.append([int(relationship[0]), int(relationship[1])])
            filtered_relationships.append(relationship)

    same_parts = common.merge_duplets(pairs)
    relationship_data = deepcopy(filtered_relationships)

    del_objects_idxs = []

    for same_part in same_parts:
        root_segment_id = same_part[0]
        
        for part_segment_id in same_part[1:]:
            segments_pd[np.where(segments_pd == part_segment_id)[0]] = root_segment_id
        
            for idx, object_data_raw in enumerate(objects[:]):
                if int(object_data_raw['id']) == part_segment_id: del_objects_idxs.append(idx)

            for idx, (sub, ob, rel_id, rel_name) in enumerate(filtered_relationships):
                sub = int(sub)
                ob = int(ob)
                rel_id = int(rel_id)

                
                if sub == part_segment_id: sub = root_segment_id
                if ob == part_segment_id: ob = root_segment_id

                if sub == ob: continue
                
                relationship_data[idx][0] = str(sub)
                relationship_data[idx][1] = str(ob)
    
    del_objects_idxs = list(set(del_objects_idxs))
    object_data = [object_data_idx for idx, object_data_idx in enumerate(objects) if idx not in del_objects_idxs]
    segment_ids_filtered = np.unique(segments_pd)
    
    return segment_ids_filtered, segments_pd, object_data, relationship_data
def process_ply_data2scan3r(scan_id, ply_data, rels_dict, objs_dict, rel2idx, cfg):
    objects_ids = [] 
    global_objects_ids = []
    objects_cat = []
    objects_attributes = []
    barry_centers = []
    
    # obj points
    points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))
    object_points = {}
    for pc_resolution in cfg.preprocess.pc_resolutions:
        object_points[pc_resolution] = []
    object_data = objs_dict['objects'] 
    for idx, object in enumerate(object_data):
        
        object_id = int(object['id'])
        object_id_for_pcl = int(object['id'])
        global_object_id = int(object['global_id'])
        obj_pt_idx = np.where(ply_data['objectId'] == object_id)
        obj_pcl = points[obj_pt_idx]
        
        if obj_pcl.shape[0] < cfg.preprocess.min_obj_points: continue
        hull = ConvexHull(obj_pcl)
        cx = np.mean(hull.points[hull.vertices,0])
        cy = np.mean(hull.points[hull.vertices,1])
        cz = np.mean(hull.points[hull.vertices,2])
        for pc_resolution in object_points.keys():
            obj_pcl = point_cloud.pcl_farthest_sample(obj_pcl, pc_resolution)
            object_points[pc_resolution].append(obj_pcl)
        barry_centers.append([cx, cy, cz])
        objects_ids.append(object_id)
        global_objects_ids.append(global_object_id)
        objects_cat.append(global_object_id)
    for pc_resolution in object_points.keys():
        object_points[pc_resolution] = np.array(object_points[pc_resolution])
    
    if len(objects_ids) < 2:
        return -1
    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_ids):
        object_id2idx[v] = index
        
    # relationship between objects
    relationships = rels_dict['relationships']
    triples = []
    pairs = []
    edges_cat = []
    for idx, triple in enumerate(relationships):
        sub = int(triple[0])
        obj = int(triple[1])
        rel_id = int(triple[2])  
        rel_name = triple[3]

        if rel_name in list(rel2idx.keys()):
            rel_id = int(rel2idx[rel_name])
            
            if sub in objects_ids and obj in objects_ids:
                assert rel_id <= len(rel2idx)
                triples.append([sub, obj, rel_id])
                edges_cat.append(rel2idx[rel_name])
                
                if triple[:2] not in pairs:
                    pairs.append([sub, obj])

    if len(pairs) == 0:
        return -1
    
    # Root Object - object with highest outgoing degree
    all_edge_objects_ids = np.array(pairs).flatten()
    root_obj_id = np.argmax(np.bincount(all_edge_objects_ids))
    root_obj_idx = object_id2idx[root_obj_id]

    # Calculate barry center and relative translation
    rel_trans = []
    for barry_center in barry_centers:
        rel_trans.append(np.subtract(barry_centers[root_obj_idx], barry_center))
    
    rel_trans = np.array(rel_trans)
    
    for i in objects_ids:
        for j in objects_ids:
            if i == j or [i, j] in pairs: continue
            triples.append([i, j, rel2idx['none']]) # supplement the 'none' relation
            pairs.append(([i, j]))
            edges_cat.append(rel2idx['none'])
    
    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    
    edges = np.stack((s, o), axis=1) 

    data_dict = {}
    data_dict['scan_id'] = scan_id
    data_dict['objects_id'] = np.array(objects_ids)
    data_dict['global_objects_id'] = np.array(global_objects_ids)
    data_dict['objects_cat'] = np.array(objects_cat)
    data_dict['triples'] = triples
    data_dict['pairs'] = pairs
    data_dict['edges'] = edges
    data_dict['obj_points'] = object_points
    data_dict['objects_count'] = len(objects_ids)
    data_dict['edges_count'] = len(edges)
    data_dict['object_id2idx'] = object_id2idx
    data_dict['object_attributes'] = objects_attributes
    data_dict['edges_cat'] = edges_cat
    data_dict['rel_trans'] = rel_trans
    data_dict['root_obj_id'] = root_obj_id
    return data_dict

def calculate_bow_node_edge_feats(data_dict_filename, rel2idx):
    data_dict = common.load_pkl_data(data_dict_filename)

    idx_2_rel = {idx : relation_name for relation_name, idx in rel2idx.items()}
    wordToIx = {}
    for key in rel2idx.keys():
        wordToIx[key] = len(wordToIx)

    edge = data_dict['edges']
    objects_ids = data_dict['objects_id']
    triples = data_dict['triples']
    edges = data_dict['edges']

    entities_edge_names = [None] * len(objects_ids)
    for idx in range(len(edges)):
        edge = edges[idx]
        entity_idx = edge[0]
        rel_name = idx_2_rel[triples[idx][2]]
        if entities_edge_names[entity_idx] is None:
            entities_edge_names[entity_idx] = [rel_name]
        else:
            entities_edge_names[entity_idx].append(rel_name)
    entity_edge_feats = None
    for entity_edge_names in entities_edge_names:
        entity_edge_feat = np.expand_dims(make_bow_vector(entity_edge_names, wordToIx), 0)
        entity_edge_feats = entity_edge_feat if entity_edge_feats is None else np.concatenate((entity_edge_feats, entity_edge_feat), axis = 0)

    data_dict['bow_vec_object_edge_feats'] = entity_edge_feats
    assert data_dict['bow_vec_object_edge_feats'].shape[0] == data_dict['objects_count']
    common.write_pkl_data(data_dict, data_dict_filename)

def make_bow_vector(sentence, word_2_idx):
    # create a vector of zeros of vocab size = len(word_to_idx)
    vec = np.zeros(len(word_2_idx))
    for word in sentence:
        if word not in word_2_idx:
            print(word)
            raise ValueError('houston we have a problem')
        else:
            vec[word_2_idx[word]]+=1
    return vec