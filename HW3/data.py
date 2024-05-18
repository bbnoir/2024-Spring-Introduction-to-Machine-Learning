from torch.utils.data import Dataset
import torch
import numpy as np

class HW2_Dataset(Dataset):
    def __init__(self, file_path):
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
        self.data = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.label = torch.tensor(data[:, 0], dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class HW2_Sample_Dataset(Dataset):
    def __init__(self, file_path):
        grid = 1000
        data = np.zeros(grid*grid*2).reshape(grid*grid, 2)
        for i in range(grid):
            for j in range(grid):
                data[i*grid+j] = [i*100.0/grid, j*100.0/grid]
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]