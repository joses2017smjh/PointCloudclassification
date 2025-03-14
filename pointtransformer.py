# modified implementation of Point Transformer architecture
# https://arxiv.org/pdf/2012.09164

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

# change to your device
device = "mps"

class TransformerLayer(nn.Module):
    def __init__(self, features):
        super(TransformerLayer, self).__init__()

        self.features = features

        # query linear transformation
        self.query = nn.Linear(in_features=features, out_features=features)

        # key linear transformation
        self.key = nn.Linear(in_features=features, out_features=features)

        # value linear transformation
        self.value = nn.Linear(in_features=features, out_features=features)

        # positional encoding
        self.pos_enc = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=features, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=1))
    
    def forward(self, x, p):

        # positional encoding
        x += self.pos_enc(p.transpose(2, 1)).transpose(2, 1)


        x_query = self.key(x)
        x_key = self.query(x)
        x_value = self.value(x)

        # attention scores
        # use scalar attention, paper uses vector attention
        x = torch.matmul(x_query, x_key.transpose(2,1)) / math.sqrt(self.features)

        # softmax on attention, apply as linear transformation
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, x_value)

        return x

class PointTransformer(nn.Module):
    def __init__(self, channels):
        super(PointTransformer, self).__init__()

        # initial linear
        self.fcl1 = nn.Linear(in_features=channels, out_features=channels)

        # multihead (4) attention
        self.head1 = TransformerLayer(channels).to(device)
        self.head2 = TransformerLayer(channels).to(device)
        self.head3 = TransformerLayer(channels).to(device)
        self.head4 = TransformerLayer(channels).to(device)

        # multihead concatenation linear transformation
        self.fcl_multihead = nn.Linear(in_features=channels*4, out_features=channels)
        
        # final linear transformation
        self.fcl2 = nn.Linear(in_features=channels, out_features=channels)

    def forward(self, x, p):

        residual = x
        
        x = self.fcl1(x)
        
        # multihead attention
        x_1 = self.head1(x.clone(), p)
        x_2 = self.head2(x.clone(), p)
        x_3 = self.head3(x.clone(), p)
        x_4 = self.head4(x.clone(), p)

        # add head changes together
        x = torch.concat([x_1, x_2, x_3, x_4], dim=-1)
        x = self.fcl_multihead(x)

        x = self.fcl2(x)
        
        # layer normalization
        x = torch.layer_norm(x, normalized_shape=[x.shape[2]])

        return x + residual, p

class TransitionDown(nn.Module):
    def __init__(self, features):
        super(TransitionDown, self).__init__()

        # shared multilayer perceptrion
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=features, out_channels=features*2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(features*2),
            nn.Conv1d(in_channels=features*2, out_channels=features*2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(features*2))
    
    def forward(self, x, p):

        # random point sampling to reduce N to N/4
        idx = torch.randint(0, x.shape[1], (int(x.shape[1]/4),))
        
        # transform each point into 2x higher feature space
        x = self.mlp(x[:,idx,:].transpose(2, 1)).transpose(2, 1)
        p = p[:,idx,:]

        return x, p

class PointTransformerBackbone(nn.Module):
    def __init__(self):
        super(PointTransformerBackbone, self).__init__()

        # initial linear
        self.init_linear = nn.Linear(in_features=3, out_features=32)
        
        # transformer/transition down blocks
        self.transformer1 = PointTransformer(channels=32).to(device)
        self.down1 = TransitionDown(features=32).to(device)
        self.transformer2 = PointTransformer(channels=64).to(device)
        self.down2 = TransitionDown(features=64).to(device)
        self.transformer3 = PointTransformer(channels=128).to(device)
        self.down3 = TransitionDown(features=128).to(device)
        self.transformer4 = PointTransformer(channels=256).to(device)
        self.down4 = TransitionDown(features=256).to(device)
        self.transformer5 = PointTransformer(channels=512).to(device)
    
    def forward(self, x):

        bs = x.shape[0]
        num_points = x.shape[1]

        p = x
        x = self.init_linear(x)

        x, p = self.transformer1(x, p)
        x, p = self.down1(x, p)
        x, p = self.transformer2(x, p)
        x, p = self.down2(x, p)
        x, p = self.transformer3(x, p)
        x, p = self.down3(x, p)
        x, p = self.transformer4(x, p)
        x, p = self.down4(x, p)
        x, p = self.transformer5(x, p)

        x = x.transpose(2, 1)

        # global average pooling to get global feature
        x = F.avg_pool1d(x, kernel_size=int(num_points/256)).view(bs, -1)

        return x

class PointTransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PointTransformerClassifier, self).__init__()
        
        self.backbone = PointTransformerBackbone().to(device)

        # fully connected layers for classification
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes))

    def forward(self, x):

        # backbone returns global feature vector
        x = self.backbone(x)
        x = self.classification_head(x)

        return x