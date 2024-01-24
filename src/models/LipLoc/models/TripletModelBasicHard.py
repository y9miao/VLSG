import timm
import torch.nn.functional as F
from torch import nn
from itertools import permutations
import numpy as np
import torch

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.device=CFG.device
        self.temperature=CFG.temperature
        self.encoder_camera = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.encoder_lidar = ImageEncoder(model_name=CFG.trained_image_model_name, pretrained=CFG.pretrained, trainable=CFG.trainable)
        self.projection_lidar = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        self.projection_camera = ProjectionHead(embedding_dim=CFG.image_embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout)
        # batch_size = CFG.batch_size
        # self.perms = np.array(list(permutations(np.arange(batch_size), 2)))
        self.loss = nn.TripletMarginLoss(margin=0.2, p=2, reduction='mean')


    def get_perms(self, size):
        perms = np.array(list(permutations(np.arange(size), 2)))
        return perms


    def forward(self, batch):
        # Getting camera Image and lidar range image Features
        batch_size = batch["camera_image"].shape[0]  # This can be different from CFG.batch_size for the last batch
        camera_features = self.encoder_camera(batch["camera_image"])
        lidar_features = self.encoder_lidar(batch["lidar_image"])
        
        # Getting camera Image and lidar range Embeddings (with same dimension)
        camera_embeddings = self.projection_camera(camera_features)
        lidar_embeddings = self.projection_lidar(lidar_features)


        ## Hard Triplet mining
        
        # Normalize the embeddings
        camera_embeddings_normalized = camera_embeddings / torch.linalg.norm(camera_embeddings, dim=1)[:, None]
        lidar_embeddings_normalized = lidar_embeddings / torch.linalg.norm(lidar_embeddings, dim=1)[:, None]

        # Multiply all the embeddings with each other : 32 x 256 * 256 x 32 -> 32 x 32
        similarity_matrix = camera_embeddings_normalized @ lidar_embeddings_normalized.T

        # Increase the diagonal values to a very high value so that they are not selected as the hardest negative
        similarity_matrix = similarity_matrix + torch.eye(batch_size).to(self.device) * 1e6
        
        # Get the hardest negative indices for each row (camera image) and column (lidar image) by selecting the minimum value
        hardest_negative_camera_indices = torch.min(similarity_matrix, dim=1).indices
        hardest_negative_lidar_indices = torch.min(similarity_matrix, dim=0).indices

        """
            l1 l2 l3
        c1  0  .8  .2
        c2  .1  0  .9
        c3
        """

        # L2C loss
        anchor = camera_embeddings
        positive = lidar_embeddings
        negative = lidar_embeddings[hardest_negative_lidar_indices]
        LossL2C = self.loss(anchor, positive, negative)

        # C2L loss
        anchor = lidar_embeddings
        positive = camera_embeddings
        negative = camera_embeddings[hardest_negative_camera_indices]
        LossC2L = self.loss(anchor, positive, negative)

        loss = ( LossL2C + LossC2L ) / 2

        return loss

        ## L2C loss
        # perms = self.get_perms(batch_size)         # 992 x 2
        # anchor = lidar_embeddings[perms[:, 0]]     # 992 x 256
        # positive = camera_embeddings[perms[:, 0]]  # 992 x 256
        # negative = camera_embeddings[perms[:, 1]]  # 992 x 256
        # LossL2C = self.loss(anchor, positive, negative)  # 992

        # print("perms", perms.shape, "anchor", anchor.shape, "positive", positive.shape, "negative", negative.shape)
        # print("LossL2C", LossL2C.shape)
        # print("Perms", perms)
        # exit(1)

        # anchor = camera_embeddings[perms[:, 0]]
        # positive = lidar_embeddings[perms[:, 0]]
        # negative = lidar_embeddings[perms[:, 1]]
        # LossC2L = self.loss(anchor, positive, negative)

        # loss = ( LossL2C + LossC2L ) / 2
        # return loss


    def get_camera_embeddings(self, batch):
        image_features = self.encoder_camera(batch["camera_image"].to(self.device))
        image_embeddings = self.projection_camera(image_features)
        return image_embeddings

    def get_lidar_embeddings(self, batch):
        image_features = self.encoder_lidar(batch["lidar_image"].to(self.device))
        image_embeddings = self.projection_lidar(image_features)
        return image_embeddings


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def get_topk(query_image_embeddings, lidar_image_embeddings, n=1):
    diff = (query_image_embeddings - lidar_image_embeddings)
    distance = torch.linalg.norm(diff, dim=1) + 1e-4
    similarity = 1 / distance
    values, indices = torch.topk(similarity, n)
    return values, indices
