import torch
import torchvision
import torch.utils.data as data

import os

class PVRConceptDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()


    def __getitem__(self, index):
        return super().__getitem__(index)


    def __len__(self):
        return 0


CONCEPT_LIST = ['TOP_LEFT','TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'IS_ONE', 'IS_TWO', 'IS_THREE', 'IS_FOUR']

def generate_PVRConceptDataset(dataset_name, dataset_path, target_path, number_of_samples=10):

    mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=False)
    cifar_dataset = None
    if dataset_name =='cifar':
        cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False)
    
    



if __name__ == "__main__":

    dataset_name = 'mnist'
    dataset_path = "F:\\pvr_dataset"
    target_path = os.path.join(dataset_path, "{dataset_name}_concept") 
    number_of_samples=10
    generate_PVRConceptDataset(dataset_name, dataset_path, target_path, 10)
