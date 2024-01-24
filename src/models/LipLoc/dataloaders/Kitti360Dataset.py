import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
import glob
from utils.model_utils import get_transforms
from utils.rangeimage_utils import loadCloudFromBinary, createRangeImage


# def build_loaders_seq(sequences, mode, CFG):
#     transforms = get_transforms(mode=mode, size=CFG.size)
#     dataset = KITTI360Dataset(
#         transforms=transforms,
#         CFG=CFG,
#         sequences=sequences,
#     )
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=CFG.batch_size,
#         num_workers=CFG.num_workers,
#         shuffle=True if mode == "train" else False,
#     )
#     return dataloader


# def build_loaders_filenames(filenames, mode, CFG):
def get_dataloader(filenames, mode, CFG):
    transforms = get_transforms(mode=mode, size=CFG.size)
    dataset = KITTI360Dataset(
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

def get_filenames(sequences, data_path, length=None):
    filenames = np.array([])
    for sequence in sorted(sequences):
        # populate timestamp files in sequence in format <Sequence digit>/<timestampfilename>
        folder = f"{data_path}/data_3d_raw/2013_05_28_drive_{sequence}_sync/velodyne_points/data/"
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
    pose_file = f"{CFG.data_path}/data_poses/2013_05_28_drive_{eval_sequence}_sync/poses.txt"

    poses = pd.read_csv(pose_file, header=None,
                        delim_whitespace=True).to_numpy()
    # X, Y, Z in Camera Init Frame | (We require Y and Z)
    translation_poses = poses[:, [4, 8, 12]]

    # Hash map of indices to poses
    indices = {}
    for i in range(poses.shape[0]):
        indices[poses[i, 0]] = poses[i, [4, 8, 12]]

    return translation_poses, indices

class KITTI360Dataset(torch.utils.data.Dataset):

    # load from sequence number
    def __init__(self, transforms, CFG, filenames=[], sequences=[]):
        if(len(filenames) != 0):
            self.data_filenames = filenames
        else:
            self.data_filenames = get_filenames(sequences, CFG.data_path)
        self.transforms = transforms
        self.CFG = CFG
        self.data_path = CFG.data_path
        self.fisheye = CFG.fisheye

    def __getitem__(self, idx):
        item = {}
        seq = self.data_filenames[idx].split('/')[0]
        instance = self.data_filenames[idx].split('/')[1]

        lidar_points = loadCloudFromBinary(f"{self.data_path}/data_3d_raw/2013_05_28_drive_{seq}_sync/velodyne_points/data/{instance}.bin") # seq ranges from 0000-0010

        if(self.fisheye):
            left_fisheye_img =  cv2.imread(f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_02/data_rgb/{instance}.png")
            right_fisheye_img =  cv2.imread(f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_03/data_rgb/{instance}.png")
            equi_width = 1400
            equi_height = 1400
            
            if left_fisheye_img is None:
                print("left image is none", f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_02/data_rgb/{instance}.png")
                left_fisheye_img = cv2.imread(f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_02/data_rgb/{str(int(instance) - 1).zfill(10)}.png")
            
            if right_fisheye_img is None:
                print("right image is none", f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_03/data_rgb/{instance}.png")
                right_fisheye_img = cv2.imread(f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_03/data_rgb/{str(int(instance) - 1).zfill(10)}.png")

            left_equi_img = self.fisheye_to_equirectangular(left_fisheye_img, equi_width, equi_height)
            right_equi_img = self.fisheye_to_equirectangular(right_fisheye_img, equi_width, equi_height)

            image = np.concatenate((left_equi_img, right_equi_img ), axis=1)
        else:
            image = cv2.imread(f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_00/data_rect/{instance}.png")

        if image is None:
            print("image is none", f"{self.data_path}/data_2d_raw/2013_05_28_drive_{seq}_sync/image_00/data_rect/{instance}.png")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['camera_image'] = torch.tensor(image).permute(2, 0, 1).float()

        # print("lidar points shape", lidar_points.shape)
        lidar_image = createRangeImage(lidar_points, self.CFG.crop) #TODO fix cropping for lidar range


        lidar_image = self.transforms(image=lidar_image)['image']
        item['lidar_image'] = torch.tensor(
            lidar_image).permute(2, 0, 1).float()
        return item

    # algo from: http://www.paulbourke.net/dome/fish2/
    def fisheye_to_equirectangular(self, fisheye_img, output_width, output_height):
        fisheye_height, fisheye_width, _ = fisheye_img.shape
        output_width = output_width * 2 # output intermediate image needs to be twice in width

        u, v = np.meshgrid(np.arange(output_width, dtype=np.float32), np.arange(output_height, dtype=np.float32))
        FOV = np.pi

        # polar angles
        theta = 2 * np.pi * (u / output_width - 0.5)
        phi =  np.pi * (v / output_height - 0.5)

        # Vector in 3d space
        x = np.multiply(np.cos(phi), np.sin(theta))
        y = np.multiply(np.cos(phi), np.cos(theta))
        z = np.sin(phi)

        # calculate fisheye angle and radius
        theta = np.arctan2(z, x)
        phi = np.arctan2(np.sqrt(np.square(x) + np.square(z)), y)
        r = fisheye_width * phi / FOV

        # pixel in fisheye space
        x_dest = 0.5 * fisheye_width + np.multiply(r , np.cos(theta))
        y_dest = 0.5 * fisheye_width + np.multiply(r , np.sin(theta))

        equirectangular_img = cv2.remap(fisheye_img, x_dest, y_dest, cv2.INTER_LINEAR)

        # clip the excess left and right
        clipStart = output_width // 4     
        equirectangular_img = equirectangular_img[:, clipStart:clipStart*3, :]

        return equirectangular_img

    def __len__(self):
        return len(self.data_filenames)

    def flush(self):
        pass
