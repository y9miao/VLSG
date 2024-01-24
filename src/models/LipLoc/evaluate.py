import math

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt
import cv2
import importlib

from dataclasses import dataclass
import tyro


##### Global Stuff ######
@dataclass
class Args:
    expid: str = 'exp_default'
    eval_sequence: str = '04'
    threshold_dist: int = 5
    
args = tyro.cli(Args)

CFG = importlib.import_module(f"config.{args.expid}").CFG

model = importlib.import_module(f"models.{CFG.model}").Model(CFG)
get_topk = importlib.import_module(f"models.{CFG.model}").get_topk
get_dataloader = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_dataloader
get_filenames = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_filenames
get_poses = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_poses
##########################


def get_lidar_image_embeddings(filenames, model):
    valid_loader = get_dataloader(filenames, mode="valid", CFG=CFG)

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.get_lidar_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def get_camera_image_embeddings(filenames, model):
    valid_loader = get_dataloader(filenames, mode="valid", CFG=CFG)

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_embeddings = model.get_camera_embeddings(batch)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def find_matches(model, lidar_embeddings, query_camera_embeddings, image_filenames, n=1):
    values, indices = get_topk(torch.unsqueeze(query_camera_embeddings,0), lidar_embeddings, n)
    matches = [image_filenames[idx] for idx in indices]
    return matches


def main():
    print(CFG.details)
    print('Evaluating On: ', args.eval_sequence)
    model_path = CFG.best_model_path

    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    all_filenames = get_filenames([args.eval_sequence], CFG.data_path, CFG.data_path_360)
    
    
    if len(args.eval_sequence) == 2:
        translation_poses = get_poses(args.eval_sequence, CFG)
    elif len(args.eval_sequence) == 4:
        translation_poses, indices = get_poses(args.eval_sequence, CFG)
        all_filenames = all_filenames[indices.astype(int)]

    # image_embeddings = get_lidar_image_embeddings([args.eval_sequence], model)
    print('Getting Lidar Embeddings...')
    lidar_embeddings = get_lidar_image_embeddings(all_filenames, model)
    lidar_embeddings = lidar_embeddings.cuda()
    

    print('Getting Camera Embeddings...')
    camera_embeddings = get_camera_image_embeddings(all_filenames, model)
    camera_embeddings = camera_embeddings.cuda()


    # Evaluation distance metric for Recall@1
    num_matches = 0
    total_queries = all_filenames.size

    # Evaluation distance metric
    diff_sum = []
    # for file in query_filenames:
    # Tqdm for progress bar
    print('Running Evaluation...')
    query_predict = []
    for i, filename in tqdm(enumerate(all_filenames)):
        
        if len(args.eval_sequence)==2:
            queryimagefilename = filename.split('/')[1]
            predictions = find_matches(model,
                                   lidar_embeddings=lidar_embeddings,
                                   query_camera_embeddings=camera_embeddings[i],
                                   image_filenames=all_filenames,
                                   n=1)
            predictedPose = int(predictions[0].split('/')[1])
            queryPose = int(queryimagefilename)
            query_predict.append([queryPose, predictedPose])
            distance = math.sqrt((translation_poses[queryPose][1] - translation_poses[predictedPose][1])**2 + (
                translation_poses[queryPose][2] - translation_poses[predictedPose][2])**2)
        
        else:
            values, pred_idx = get_topk(torch.unsqueeze(camera_embeddings[i],0), lidar_embeddings, 1)
            predIdx = pred_idx[0]
            queryIdx = i
            distance = math.sqrt((translation_poses[queryIdx][1] - translation_poses[predIdx][1])**2 + (
                                translation_poses[queryIdx][2] - translation_poses[predIdx][2])**2)
        # diff_sum.append(distance)
        if(distance < args.threshold_dist):
            num_matches += 1

    # print(np.mean(diff_sum))
    recall = num_matches/total_queries
    print("Recall@1: ", recall)
    query_predict = np.array([query_predict])
    np.save(f'data/eval_predictions_{args.eval_sequence}.np', query_predict)


if __name__ == "__main__":
    main()
