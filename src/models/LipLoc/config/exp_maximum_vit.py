import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    data_path = "/ssd_scratch/saishubodh/cross_modal_kitti_data/padfoot7/cross-modal-place-recognition/data/SEMANTIC-KITTI-DATASET/sequences/"
    debug = False
    train_sequences =  ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    expdir = f"data/{expid}/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    batch_size = 32
    num_workers = 4 # 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = 'vit_small_patch16_224' #'resnet50' 
    image_embedding_dim = 2048
    max_length = 200
    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0
    # image size
    size = 224
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    crop = True
    dataloader = "KittiDataset"
    model = "CLIPModelV1_vit"

    logdir = f"data/{expid}/log/"

    details = f"Exp Id: {expid} \nTraining on: {train_sequences} \nBatch Size: {batch_size}"
