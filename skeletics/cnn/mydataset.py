import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, data_path, label_path):
        data, labels = np.load(data_path), np.load(label_path)
        #chosen_labels = [1,25,30,36,40,47,10,13,17,20]
        #chosen_indices = np.where(np.isin(labels, chosen_labels))[0]
        #data, labels = data[chosen_indices], labels[chosen_indices]
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

