import cv2
import numpy as np
import torch
import glob
import pandas as pd
from PIL import Image
from utils.model_utils import get_transforms
from utils.rangeimage_utils import loadCloudFromBinary, createRangeImage

# KITTI 360 data path : /data_2d_raw /data_3d_raw
# KITTI data path: / 00 01 ...


def get_dataloader(filenames, mode, CFG):
    transforms = get_transforms(mode=mode, size=CFG.size)
    dataset = KITTIBothDataset(
        transforms=transforms,
        CFG=CFG,
        filenames=filenames,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def get_filenames(sequences, data_path, data_path_360, length=None):
    filenames = np.array([])
    for sequence in sorted(sequences):
        # populate timestamp files in sequence in format <Sequence digit>/<timestampfilename>

        if len(sequence) == 2:
            folder = f"{data_path}/{sequence}/velodyne/"
            files = glob.glob(folder+'*bin')
            image_arr = np.arange(0, len(files)).astype(str)
            image_ids = np.char.zfill(image_arr,6)
            image_ids = np.core.defchararray.add(f"{sequence}/", image_ids)
            filenames = np.append(filenames,image_ids)
        elif len(sequence) == 4:
            folder = f"{data_path_360}/data_3d_raw/2013_05_28_drive_{sequence}_sync/velodyne_points/data/"
            files = glob.glob(folder+'*bin')
            image_arr = np.arange(0, len(files)).astype(str) #TODO seq 0002 does not start with 0000000000
            image_ids = np.char.zfill(image_arr,10)
            image_ids = np.core.defchararray.add(f"{sequence}/", image_ids)
            filenames = np.append(filenames,image_ids)

        if(length is not None):
            filenames = filenames[0:length]
    return filenames

def get_poses(eval_sequence, CFG):
     # get all poses in training
    if len(eval_sequence) == 2:
        pose_file = CFG.data_path + '/' + eval_sequence + '/poses.txt'

        poses = pd.read_csv(pose_file, header=None,
                        delim_whitespace=True).to_numpy()

        translation_poses = poses[:, [3, 7, 11]]
    
        return translation_poses

    elif len(eval_sequence) == 4:
        pose_file = f"{CFG.data_path_360}/data_poses/2013_05_28_drive_{eval_sequence}_sync/poses.txt"
        poses = pd.read_csv(pose_file, header=None,
                        delim_whitespace=True).to_numpy()
        # X, Y, Z in Camera Init Frame | (We require Y and Z)
        translation_poses = poses[:, [4, 8, 12]]
        # Hash map of indices to poses
        indices = {}
        for i in range(poses.shape[0]):
            indices[poses[i, 0]] = poses[i, [4, 8, 12]]
        
        return translation_poses, indices


class KITTIBothDataset(torch.utils.data.Dataset):

    # load from sequence number
    def __init__(self, transforms, CFG, filenames=[], sequences=[]):
        if(len(filenames) != 0):
            self.data_filenames = filenames
        else:
            self.data_filenames = get_filenames(sequences, CFG.data_path, CFG.data_path_360)
        self.transforms = transforms

        self.data_path = CFG.data_path
        self.data_path_360 = CFG.data_path_360
        self.CFG = CFG

    def __getitem__(self, idx):
        item = {}
        seq = self.data_filenames[idx].split('/')[0]
        instance = self.data_filenames[idx].split('/')[1]

        if len(seq) == 2:
            image1 = cv2.imread(f"{self.data_path}/{seq}/image_2/{instance}.png")
            lidar_points = loadCloudFromBinary(
                f"{self.data_path}/{seq}/velodyne/{instance}.bin")
        elif len(seq) == 4:
            image1 = cv2.imread(f"{self.data_path_360}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_00/data_rect/{instance}.png")
            lidar_points = loadCloudFromBinary(f"{self.data_path_360}/data_3d_raw/2013_05_28_drive_{seq}_sync/velodyne_points/data/{instance}.bin") 
        
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['camera_image'] = torch.tensor(image).permute(2, 0, 1).float()
        lidar_image = createRangeImage(lidar_points, self.CFG.crop)

        lidar_image = self.transforms(image=lidar_image)['image']
        item['lidar_image'] = torch.tensor(
            lidar_image).permute(2, 0, 1).float()
        return item

    def __len__(self):
        return len(self.data_filenames)

    def flush(self):
        pass
