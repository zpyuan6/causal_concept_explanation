from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import matplotlib.pylab as plt

class NeuronalAbstractionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_path:str, model_name:str, train_or_val:str, concept_remove_y:bool=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        data_input_file = np.load(os.path.join(dataset_path, f"{model_name}_{train_or_val}_neuronal_abstractions.npy"), allow_pickle=True)
        data_output_file = np.load(os.path.join(dataset_path, f"{model_name}_{train_or_val}_concepts.npy"), allow_pickle=True)

        self.x = torch.from_numpy(data_input_file)

        self.y = torch.from_numpy(data_output_file).to(torch.int64)

        self.concept_remove_y = concept_remove_y

        print(f"Load Neuronal Abstraction Dataset {self.x.shape}, {self.y.shape}")

    def get_input_shape(self):
        return self.x.shape[1:]

    def get_concept_dims(self):
        return (torch.max(self.y, dim=0)[0] + 1).tolist()
    
    def get_concept_labels(self):
        return self.y
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.concept_remove_y:
            return self.x[idx], self.y[idx][-1], self.y[idx][:-1]
        else:
            return self.x[idx], self.y[idx][-1], self.y[idx]