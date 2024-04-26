import os 
import os.path as osp
from re import sub
from sympy import root
from tqdm import tqdm
import numpy as np 
from scipy.spatial import ConvexHull
import argparse
import random
import sys
from yacs.config import CfgNode as CN
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common, point_cloud, scan3r
from configs import config, update_config


def process_scan(data_dir, rel_data, obj_data, cfg, rel2idx, rel_transforms = None):
    scan_id = rel_data['scan']

    if len(rel_data['relationships']) == 0:
        return -1

    objects_ids = [] 
    global_objects_ids = []
    objects_cat = []
    objects_attributes = []
    barry_centers = []

    ply_data_npy_file = osp.join(data_dir, 'scenes', scan_id, 'data.npy')
    ply_data = None
    if( osp.isfile(ply_data_npy_file)): 
        ply_data = np.load(ply_data_npy_file)
    else:
        # Load scene pcl
        ply_data = scan3r.save_ply_data(osp.join(data_dir, 'scenes'), 
            scan_id, "labels.instances.align.annotated.v2.ply", ply_data_npy_file)
        
    points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))

    object_points = {}
    for pc_resolution in cfg.preprocess.pc_resolutions:
        object_points[pc_resolution] = []

    object_data = obj_data['objects'] 
    
    for idx, object in enumerate(object_data):
        if not cfg.use_predicted : attribute = [item for sublist in object['attributes'].values() for item in sublist]

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
        if not cfg.use_predicted : objects_attributes.append(attribute)
    
    for pc_resolution in object_points.keys():
        object_points[pc_resolution] = np.array(object_points[pc_resolution])
    
    if len(objects_ids) < 2:
        return -1
    
    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_ids):
        object_id2idx[v] = index

    relationships = rel_data['relationships']
    triples = []
    pairs = []
    edges_cat = []

    for idx, triple in enumerate(relationships):
        sub = int(triple[0])
        obj = int(triple[1])
        rel_id = int(triple[2])  
        rel_name = triple[3] if rel_transforms is None else rel_transforms[triple[3]]

        if rel_name in list(rel2idx.keys()):
            rel_id = int(rel2idx[rel_name])
            
            if sub in objects_ids and obj in objects_ids:
                if rel_name == 'inside':
                    assert False
                
                assert rel_id <= len(rel2idx)
                triples.append([sub, obj, rel_id])
                edges_cat.append(rel2idx[rel_name])
                
                if triple[:2] not in pairs:
                    pairs.append([sub, obj])

    # if len(pairs) == 0:
    #     return -1

    # Root Object - object with highest outgoing degree
    all_edge_objects_ids = np.array(pairs).flatten()
    if len(all_edge_objects_ids) == 0:
        # randomly select a root object
        root_obj_id = random.choice(objects_ids)
    else:
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

def process_data(cfg, rel2idx, rel_transforms = None, mode = 'orig', split = 'train', data_file = 'data'):
    use_predicted = cfg.use_predicted
    scan_type = cfg.scan_type
    anchor_type_name = cfg.preprocess.anchor_type_name
    out_dirname = '' if scan_type == 'scan' else 'out'
    out_dirname = osp.join(out_dirname, 'predicted') if use_predicted else out_dirname
    data_dir = osp.join(cfg.data.root_dir, out_dirname)
    data_write_dir = osp.join(data_dir, 'files', mode)
    
    common.ensure_dir(data_write_dir)
    common.ensure_dir(osp.join(data_write_dir, data_file))
    
    print('[INFO] Processing subscans from {} split'.format(split))
    
    rel_json_filename = 'relationships.json' if scan_type == 'scan' else 'relationships_subscenes_{}.json'.format(split)
    obj_json_filename = 'objects.json' if scan_type == 'scan' else 'objects_subscenes_{}.json'.format(split)
    resplit = 'resplit_' if cfg.data.resplit else ''
    scan_ids_filename = '{}_{}scans.txt'.format(split, resplit) if scan_type == 'scan' else '{}_scans_subscenes.txt'.format(split)

    rel_json = common.load_json(osp.join(data_dir, 'files', rel_json_filename))['scans']
    obj_json = common.load_json(osp.join(data_dir, 'files', obj_json_filename))['scans']

    subscan_ids_generated = np.genfromtxt(osp.join(data_dir, 'files', scan_ids_filename), dtype=str)  
    subscan_ids_processed = []
    
    subRescan_ids_generated = []
    scans_dir = cfg.data.root_dir
    scans_files_dir = osp.join(scans_dir, 'files')
    all_scan_data = common.load_json(osp.join(scans_files_dir, '3RScan.json'))
    # get rescans
    for scan_data in all_scan_data:
        ref_scan_id = scan_data['reference']
        if ref_scan_id in subscan_ids_generated:
            rescan_ids = [scan['reference'] for scan in scan_data['scans']]
            subRescan_ids_generated += rescan_ids + [ref_scan_id]
    subscan_ids_generated = subRescan_ids_generated

    for subscan_id in tqdm(subscan_ids_generated[:]):
        obj_data = [obj_data for obj_data in obj_json if obj_data['scan'] == subscan_id][0]
        rel_data = [rel_data for rel_data in rel_json if rel_data['scan'] == subscan_id][0]
        data_dict = process_scan(data_dir, rel_data, obj_data, cfg, rel2idx, rel_transforms=rel_transforms)
        
        if type(data_dict) == int: continue

        common.write_pkl_data(data_dict, osp.join(data_write_dir, data_file, data_dict['scan_id'] + '.pkl'))
        subscan_ids_processed.append(subscan_id)
    
    subscan_ids = np.array(subscan_ids_processed) 

    return data_dir, data_write_dir, mode, subscan_ids

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

def calculate_bow_node_edge_feats(data_write_dir, rel2idx, scan_ids, data_file):
    print('[INFO] Starting BOW Feature Calculation For Node Edge Features...')
    
    scan_ids = sorted([scan_id for scan_id in scan_ids])

    idx_2_rel = {idx : relation_name for relation_name, idx in rel2idx.items()}
    
    wordToIx = {}
    for key in rel2idx.keys():
        wordToIx[key] = len(wordToIx)

    print('[INFO] Size of Node Edge Vocabulary - {}'.format(len(wordToIx)))
    print('[INFO] Generated Vocabulary, Calculating BOW Features...')
    for scan_id in scan_ids:
        data_dict_filename = osp.join(data_write_dir, data_file, '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        
        edge = data_dict['edges']
        objects_ids = data_dict['objects_id']
        triples = data_dict['triples']
        edges = data_dict['edges']

        entities_edge_names = [None] * len(objects_ids)
        for idx in range(len(edges)):
            edge = edges[idx]
            entity_idx = edge[0]
            rel_name = idx_2_rel[triples[idx][2]]

            if rel_name == 'inside':
                print(scan_id)

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
    
    print('[INFO] Completed BOW Feature Calculation For Node Edge Features.')

def calculate_bow_node_attr_feats(data_write_dir, scan_ids, data_file, word_2_ix):
    print('[INFO] Starting BOW Feature Calculation For Node Attribute Features...')
    
    scan_ids = sorted([scan_id for scan_id in scan_ids])
    for scan_id in tqdm(scan_ids):
        data_dict_filename = osp.join(data_write_dir, data_file, '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        attributes = data_dict['object_attributes']

        for object_attr in attributes:
            for attr in object_attr:
                if attr not in word_2_ix:
                    word_2_ix[attr] = len(word_2_ix)

    print('[INFO] Size of Node Attribute Vocabulary - {}'.format(len(word_2_ix)))
    print('[INFO] Generated Vocabulary, Calculating BOW Features...')

    for scan_id in scan_ids:
        data_dict_filename = osp.join(data_write_dir, data_file, '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        attributes = data_dict['object_attributes']

        bow_vec_attrs = None
        for object_attr in attributes:
            bow_vec_attr = np.expand_dims(make_bow_vector(object_attr, word_2_ix), 0)
            bow_vec_attrs = bow_vec_attr if bow_vec_attrs is None else np.concatenate((bow_vec_attrs, bow_vec_attr), axis = 0)
        
        data_dict['bow_vec_object_attr_feats'] = bow_vec_attrs
        assert bow_vec_attrs.shape[0] == data_dict['objects_count']
        common.write_pkl_data(data_dict, data_dict_filename)
    
    print('[INFO] Completed BOW Feature Calculation For Node Attribute Features.')
    
def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', help='Seed for random number generator')
    return parser.parse_known_args()

if __name__ == '__main__':
    # get arguments
    args, _ = parse_args()
    cfg_file = args.config
    split = args.split
    
    # cfg = CN()
    # cfg.defrost()
    # cfg.set_new_allowed(True)
    # cfg.merge_from_file(cfg_file)
    cfg = update_config(config, cfg_file, ensure_dir = False)

    random.seed(cfg.seed)
    root_dir = cfg.data.root_dir
    REL2IDX_SCANNET8 = common.name2idx(osp.join(root_dir, 'files/scannet8_relationships.txt'))
    REL2IDX_SCANNET41 = common.name2idx(osp.join(root_dir, 'files/relationships.txt'))
    CLASS2IDX_SCANNET20 = common.name2idx(osp.join(root_dir, 'files/scannet20_classes.txt'))
    rel2idx_8 = REL2IDX_SCANNET8
    rel2idx_41 = REL2IDX_SCANNET41
    
    # whether use 8  rel or 40 rel
    if cfg.model.rel_dim == 9:
        rel2idx = rel2idx_8
        rel_tran_file = osp.join(cfg.data.root_dir, 'files/rel41To9.pkl')
        rel_transform = common.load_pkl_data(rel_tran_file)
        data_file = 'data_rel9'
    else:
        rel2idx = rel2idx_41
        rel_transform = None
        data_file = 'data'
    data_dir, data_write_dir, mode, scan_ids = process_data(cfg, rel2idx, rel_transform, split=split, data_file=data_file)

    data_write_dir = osp.join(root_dir, 'files', 'orig')
    common.ensure_dir(data_write_dir)
    # obj attributes
    OBJ_ATTR_FILENAME = osp.join(root_dir, 'files', 'obj_attr.pkl')
    word_2_ix = common.load_pkl_data(OBJ_ATTR_FILENAME)
    if not cfg.use_predicted : calculate_bow_node_attr_feats(data_write_dir, 
                                        scan_ids, data_file= data_file, word_2_ix = word_2_ix)
    # relationships
    calculate_bow_node_edge_feats(data_write_dir, rel2idx, scan_ids, data_file= data_file)