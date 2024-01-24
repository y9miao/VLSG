import cv2
import numpy as np
import torch
from PIL import Image
from utils.model_utils import get_filenames, get_transforms
from utils.rangeimage_utils import loadCloudFromBinary, createRangeImage


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
        
        # Random shuffle of the data_filenames_negatves of the copy of the list
        self.data_filenames_negatives = self.data_filenames.copy()
        np.random.shuffle(self.data_filenames_negatives)

        self.transforms = transforms

        self.data_path = CFG.data_path
        self.CFG = CFG

    def __getitem__(self, idx):
        item = {}
        seq = self.data_filenames[idx].split('/')[0]
        instance = self.data_filenames[idx].split('/')[1]
        image1 = cv2.imread(f"{self.data_path}/{seq}/image_2/{instance}.png")
        lidar_points = loadCloudFromBinary(
            f"{self.data_path}/{seq}/velodyne/{instance}.bin")

        # Negative
        seq_neg = self.data_filenames_negatives[idx].split('/')[0]
        instance_neg = self.data_filenames_negatives[idx].split('/')[1]
        image1_neg = cv2.imread(f"{self.data_path}/{seq_neg}/image_2/{instance_neg}.png")
        lidar_points_neg = loadCloudFromBinary(f"{self.data_path}/{seq_neg}/velodyne/{instance_neg}.bin")

        # Don't Stitch
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['camera_image'] = torch.tensor(image).permute(2, 0, 1).float()
        lidar_image = createRangeImage(lidar_points, self.CFG.crop)

        lidar_image = self.transforms(image=lidar_image)['image']
        item['lidar_image'] = torch.tensor(
            lidar_image).permute(2, 0, 1).float()


        # Negative
        image_neg = cv2.cvtColor(image1_neg, cv2.COLOR_BGR2RGB)
        image_neg = self.transforms(image=image_neg)['image']
        item['camera_image_neg'] = torch.tensor(image_neg).permute(2, 0, 1).float()

        lidar_image_neg = createRangeImage(lidar_points_neg, self.CFG.crop)
        lidar_image_neg = self.transforms(image=lidar_image_neg)['image']
        item['lidar_image_neg'] = torch.tensor(lidar_image_neg).permute(2, 0, 1).float()
        
        return item

    def __len__(self):
        return len(self.data_filenames)

    def flush(self):
        pass
