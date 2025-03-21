# modified implementation of PointNet architecture
# https://arxiv.org/pdf/1612.00593

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# change to your device
device = "cuda"

class PointNetClassificationLoss(nn.Module):
    def __init__(self, reg_weight):
        super(PointNetClassificationLoss, self).__init__()

        # regularization term
        self.reg_weight = reg_weight

        # based on cross entropy loss for classification
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, outputs, labels, A):

        bs = A.shape[0]

        loss = self.cross_entropy(outputs, labels)

        # calculate loss taking into account orthogonality of feature transformation
        I = torch.eye(64).repeat(bs, 1, 1).view(bs, 64, 64).to(device)
        loss += self.reg_weight*torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))/bs

        return loss


class Transformer(nn.Module):
    def __init__(self, features, identity):
        super(Transformer, self).__init__()

        self.features = features
        self.identity = identity

        # shared multilayer perceptron
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024))

        # linear to create matrix
        self.fcl = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=features*features))
    
    def forward(self, x):

        bs = x.shape[0]
        num_points = x.shape[2]

        # shared multilayer perceptron, into global max pooling to create feature vector
        x = self.mlp(x)
        x = F.max_pool1d(x, kernel_size=num_points).view(bs, -1)
        x = self.fcl(x)
        
        # create matrix from feature vector
        x = x.view(-1, self.features, self.features)

        # initialize as identity (for larger transformation), refine using network
        if self.identity:
            x += torch.eye(self.features, requires_grad=True).repeat(bs, 1, 1).to(device)

        return x


class PointNetBackbone(nn.Module):
    def __init__(self, num_classes):
        super(PointNetBackbone, self).__init__()

        # first T-Net
        self.tnet1 = Transformer(features=3, identity=False).to(device)

        # shared multilayer perceptron
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64))
    
        # second T-Net
        self.tnet2 = Transformer(features=64, identity=True).to(device)

        # shared multilayer perceptron
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(1024))
    
    def forward(self, x):

        bs = x.shape[0]
        num_points = x.shape[2]

        input_transform = self.tnet1(x)
        
        # apply first learned T-Net
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)
        x = self.mlp1(x)

        feature_transform = self.tnet2(x)

        # apply second learned T-Net
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        features = x
        x = self.mlp2(x)

        # create global feature vector
        x = F.max_pool1d(x, kernel_size=num_points).view(bs, -1)

        return x, feature_transform, features


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PointNetClassifier, self).__init__()

        self.backbone = PointNetBackbone(num_classes=num_classes).to(device)

        # fully connected layers for classification
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=num_classes))

    def forward(self, x):

        x = x.transpose(2, 1)

        # backbone returns global feature vector, larger T-Net (used for regularization), and point features
        x, feature_transform, features = self.backbone(x)
        x = self.classification_head(x)

        return x, feature_transform