from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import matplotlib.pylab as plt

class CausalPVRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_path:str, dataset_name:str, train_or_val:str, num_class:int):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if dataset_name not in ["mnist", "cifar"]:
            raise Exception(f"Can not find dataset {dataset_name}.")

        file = open(os.path.join(dataset_path, f"{train_or_val}_{dataset_name}_pvr.txt"),'rb')    

        x, y = pickle.load(file)

        self.x = torch.from_numpy(np.array(x))/255

        if self.x.shape[1] != 3:
            self.x = self.x.repeat(1,3,1,1)

        # print(self.x.shape, torch.unsqueeze(torch.from_numpy(np.array(y)),1).dtype)
        # self.y = torch.nn.functional.one_hot(torch.unsqueeze(torch.from_numpy(np.array(y)),1).to(torch.int64) , num_class)
        self.y = torch.from_numpy(np.array(y)).to(torch.int64)

        print(self.x.shape, self.y.shape)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx][-1], self.y[idx]


if __name__=="__main__":

    batch_size = 1
    dataset_path = "F:\pvr_dataset\cifar_pvr"
    dataset_name = "cifar"
    num_class=10

    train_dataset = PVRDataset(dataset_path, dataset_name, "train", num_class)
    val_dataset = PVRDataset(dataset_path, dataset_name, "val", num_class)

    x_t,y_t = train_dataset[0]
    
    print(y_t)
    img = np.transpose(np.squeeze(x_t.numpy()),(1,2,0))
    plt.imshow(img)
    plt.show()

    x_t,y_t = val_dataset[0]
    print(y_t)
    img = np.transpose(np.squeeze(x_t.numpy()),(1,2,0))
    plt.imshow(img)
    plt.show()