import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda"





class PointCloudTransformerBackbone(nn.Module):
    def __init__(self, num_classes):
        super(PointCloudTransformerBackbone, self).__init__()

        # todo
    
    def forward(self, x):

        # todo


class PointCloudTransformerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PointCloudTransformerClassifier, self).__init__()

       # todo

    def forward(self, x):

        # todo