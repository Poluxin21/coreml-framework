import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, num_features=3, num_classes=2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.X = torch.randn(num_samples, num_features)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_data(num_samples=1000, num_features=3, num_classes=2):
    dataset = SimpleDataset(num_samples, num_features, num_classes)
    dataset.num_features = num_features
    return dataset

def generate_synthetic_data(num_samples=1000, num_features=10, output_file='data.csv'):
    data = np.random.rand(num_samples, num_features)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
    df['label'] = np.random.randint(0, 2, size=num_samples)
    df.to_csv(output_file, index=False)
    print(f"Generated synthetic data and saved to {output_file}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    generate_synthetic_data(output_file=output_path)