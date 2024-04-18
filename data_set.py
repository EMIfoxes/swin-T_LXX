import torch
from torch.utils.data import Dataset
import pandas as pd
import hdf5storage
import matplotlib.pyplot as plt
class CustomCWRUDataset(Dataset):
    def __init__(self, data_pd, transform=None):
        self.data_pd = data_pd
        self.transform = transform

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
        file_path = self.data_pd.iloc[idx]['data']
        label = int(self.data_pd.iloc[idx]['label'])
        data = hdf5storage.loadmat(file_path)['single_data']#提取数据并转置
        data = torch.tensor(data).float().reshape(-1)
        if self.transform:
            data = self.transform(data)
        return data, label

import os
import pandas as pd
from sklearn.model_selection import train_test_split

class CWRU(object):
    def __init__(self, root_dir, test_size=0.2,sample=None, transform=None):
        self.root_dir = root_dir
        self.test_size = test_size
        self.sample = sample
        self.data_pd = self.load_cwru_data()
        self.transform = transform


    def load_cwru_data(self):
        data = {'data': [], 'label': []}
        num=0

        for class_label, class_name in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    data['data'].append(file_path)
                    data['label'].append(class_label)
                    num += 1
                    if num==self.sample:
                        num=0
                        break

        return pd.DataFrame(data)

    def train_test_split_order(self):
        train_pd, test_pd, _, _ = train_test_split(
            self.data_pd,
            self.data_pd['label'],
            test_size=self.test_size,
            stratify=self.data_pd['label'],  # Ensure stratified split based on labels
            random_state=123  # Set a seed for reproducibility
        )
        train_dataset = CustomCWRUDataset(train_pd)
        test_dataset = CustomCWRUDataset(test_pd)
        return train_dataset, test_dataset