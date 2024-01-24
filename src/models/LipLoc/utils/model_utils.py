import glob
import albumentations as A
import numpy as np


def get_filenames(sequences, data_path, length=None):
    filenames = np.array([])
    for sequence in sorted(sequences):
        # populate timestamp files in sequence in format <Sequence digit>/<timestampfilename>
        folder = f"{data_path}/{sequence}/velodyne/"
        files = glob.glob(folder+'*bin')
        image_arr = np.arange(0, len(files)).astype(str)
        image_ids = np.char.zfill(image_arr,6)
        image_ids = np.core.defchararray.add(f"{sequence}/", image_ids)
        filenames = np.append(filenames,image_ids)

        if(length is not None):
            filenames = filenames[0:length]
    return filenames

def get_transforms(mode, size):
    if mode == "train":
        return A.Compose(
            [   
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
                A.CoarseDropout(always_apply=False, p=0.9, max_holes=5, max_height=100, max_width=100, min_height=50, min_width=50, fill_value=(0, 0, 0), mask_fill_value=None),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
