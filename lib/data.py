import torch
from torch.utils.data import Dataset, DataLoader

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

class downscalingDataset(Dataset):

    '''
    Basic Dataset class
    '''

    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :, :]
        y = self.y[idx, :]

        return x, y
    
class downscalingDatasetSamples(Dataset):

    '''
    Basic Dataset class
    '''

    def __init__(self, x, y=[]):
        self.x = torch.from_numpy(x).float()
        self.y = [torch.from_numpy(elem).float() for elem in y] 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx, :, :, :]
        y = sample(self.y)[idx, :]

        return x, y