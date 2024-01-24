import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    data_path = "../data/SEMANTIC-KITTI-DATASET/sequences/" #TODO: Set path
    data_path_360 = "../data/KITTI360_download/FINAL/KITTI-360" #TODO: Set path
    debug = False
    train_sequences =  ["00", "01", "02", "03", "04", "05", "06", "07", "11", "12", "13", "15", "16", "17", "18", "19", "20", "21", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010"]
    expdir = f"data/{expid}/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    batch_size = 32
    num_workers = 10
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_image_model_name = 'resnet50'
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
    
    # Cropping
    crop = True
    crop_distance=False
    distance_threshold=100
    
    dataloader = "KittiBothDataset"
    model = "CLIPModelV1"

    logdir = f"data/{expid}/log/"

    details = f"Exp Id: {expid} \nTraining on: {train_sequences} \nBatch Size: {batch_size}"
