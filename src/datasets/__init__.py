from .scan3r_objpair_XTAE_SGI import PatchObjectPairXTAESGIDataSet

def get_dataset(dataset_name):
    if dataset_name == 'Scan3R':
        return PatchObjectPairXTAESGIDataSet
    else:
        raise NotImplementedError