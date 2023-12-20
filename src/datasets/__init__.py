from .scan3r_obj_pair import PatchObjectPairDataSet

def get_dataset(dataset_name):
    if dataset_name == 'Scan3R':
        return PatchObjectPairDataSet
    else:
        raise NotImplementedError