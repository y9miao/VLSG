import torch
from pathlib import Path

class CFG:
    expid = Path(__file__).stem
    debug = False

    # data loader 
    dataloader = "Kitti360Dataset"
    fisheye = False
    data_path = "../KITTI360_download/FINAL/KITTI-360" #TODO: Set path
    train_sequences = ["0003", "0004", "0005", "0006", "0007", "0009", "0010"]

    logdir = f"data/{expid}/log/"

    
    expdir = f"data/{expid}/"
    best_model_path = f"{expdir}best.pth"
    final_model_path = f"{expdir}model.pth"
    batch_size = 16
    num_workers = 8
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
    wandbMode = "disabled" #online or disabled
    model = "CLIPModelV1"

    # Cropping
    crop = True
    crop_distance=True
    distance_threshold=50

    details = f"Exp Id: {expid} \nTraining Sequences: {train_sequences} \nBatch Size: {batch_size}"