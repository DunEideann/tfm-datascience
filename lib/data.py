import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

class downscalingDataset(Dataset):

    '''
    Basic Dataset class
    '''

    def __init__(self, x, y):
        #mask = ~np.isnan(x).any(axis=(1, 2, 3)) & ~np.isnan(y).any(axis=1)
        self.x = torch.from_numpy(x).float()#x[mask]).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :, :]
        y = self.y[idx, :]

        return x, y
    
class downscalingDatasetEnsemble(Dataset):

    '''
    Basic Dataset class
    '''

    def __init__(self, x, y=[]):
        # mask = ~np.isnan(x).any(axis=(1, 2, 3))  # Mask for x
        # for elem in y:
        #     mask &= ~np.isnan(elem).any(axis=1)

        self.x = torch.from_numpy(x).float()
        self.y = [torch.from_numpy(elem).float() for elem in y] 
        #self.yIndex = []

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :, :]

        y_index = torch.randint(0, len(self.y), (1,)).item()  # Seleccionamos un índice aleatorio de self.y
        y_random = self.y[y_index]
        y = y_random[idx, :]
        #self.yIndex.append(y_index)

        return x, y
    