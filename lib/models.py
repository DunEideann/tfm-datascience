import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DeepESD(torch.nn.Module):

    '''
    DeepESD model as defined in
    https://gmd.copernicus.org/articles/15/6747/2022/
    '''
    
    def __init__(self, spatial_x_dim, out_dim,
                 channelsInputLayer, channelsLastLayer):

        super(DeepESD, self).__init__()

        self.channelsInputLayer = channelsInputLayer
        self.channelsLastLayer = channelsLastLayer

        self.conv1 = torch.nn.Conv2d(in_channels=self.channelsInputLayer,
                                     out_channels=50,
                                     kernel_size=3,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=50,
                                     out_channels=25,
                                     kernel_size=3,
                                     padding=1)

        self.conv3 = torch.nn.Conv2d(in_channels=25,
                                     out_channels=self.channelsLastLayer,
                                     kernel_size=3,
                                     padding=1)

        self.out = torch.nn.Linear(in_features=spatial_x_dim[0] * spatial_x_dim[1] * self.channelsLastLayer,
                                    out_features=out_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim = 1)

        out = self.out(x)

        return out

