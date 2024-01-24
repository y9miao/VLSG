import cv2
import numpy as np
import torch
from PIL import Image
from utils.model_utils import get_filenames, get_transforms
from utils.rangeimage_utils import loadCloudFromBinary, createRangeImage


# def build_loaders_seq(sequences, mode, CFG):
#     transforms = get_transforms(mode=mode, size=CFG.size)
#     dataset = CILPDataset(
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


def get_dataloader(filenames, mode, CFG):
    transforms = get_transforms(mode=mode, size=CFG.size)
    dataset = CILPDataset(
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


class CILPDataset(torch.utils.data.Dataset):

    # load from sequence number
    def __init__(self, transforms, CFG, filenames=[], sequences=[]):
        if(len(filenames) != 0):
            self.data_filenames = filenames
        else:
            self.data_filenames = get_filenames(sequences, CFG.data_path)
        self.transforms = transforms

        self.data_path = CFG.data_path
        self.CFG = CFG

    def __getitem__(self, idx):
        item = {}
        seq = self.data_filenames[idx].split('/')[0]
        instance = self.data_filenames[idx].split('/')[1]
        image1 = cv2.imread(f"{self.data_path}/{seq}/image_2/{instance}.png")
        # image2 = cv2.imread(f"{CFG.rootPath}/{seq}/image_3/{instance}.png")
        lidar_points = loadCloudFromBinary(
            f"{self.data_path}/{seq}/velodyne/{instance}.bin")

        # Don't Stitch
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['camera_image'] = torch.tensor(image).permute(2, 0, 1).float()
        lidar_image = createRangeImage(lidar_points, self.CFG.crop, self.CFG.crop_distance, self.CFG.distance_threshold)

        # ## Testing
        # from matplotlib import pyplot as plt
        # plt.imsave(f'data/corrected_li_{idx}.png', lidar_image)
        # plt.clf()
        # plt.imsave(f'data/corrected_ci_{idx}.png', image1)
        # plt.clf()

        lidar_image = self.transforms(image=lidar_image)['image']
        item['lidar_image'] = torch.tensor(
            lidar_image).permute(2, 0, 1).float()
        return item

    def __len__(self):
        return len(self.data_filenames)

    def flush(self):
        pass
