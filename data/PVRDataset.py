from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
import random

class CausalPVRDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_path:str, train_or_val:str):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        data_input_file = np.load(os.path.join(dataset_path, f"{train_or_val}_image_casaul_pvr.npy"), allow_pickle=True).transpose(0,3,1,2)
        data_output_file = np.load(os.path.join(dataset_path, f"{train_or_val}_concept_casaul_pvr.npy"), allow_pickle=True)

        self.x = torch.from_numpy(data_input_file)/255

        if self.x.shape[1] != 3:
            self.x = self.x.repeat(1,3,1,1)

        # print(self.x.shape, torch.unsqueeze(torch.from_numpy(np.array(y)),1).dtype)
        # self.y = torch.nn.functional.one_hot(torch.unsqueeze(torch.from_numpy(np.array(y)),1).to(torch.int64) , num_class)
        self.y = torch.from_numpy(data_output_file).to(torch.int64)

        print(self.x.shape, self.y.shape)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx][-1], self.y[idx]


if __name__=="__main__":

    batch_size = 1
    dataset_path = "F:\\causal_pvr_v2\\chain"
    dataset_name = "mnist"
    num_class=10

    # train_dataset = CausalPVRDataset(dataset_path, dataset_name, "train", num_class)
    # val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val", num_class)

    train_dataset = CausalPVRDataset(dataset_path, "train")
    # val_dataset = CausalPVRDataset(dataset_path, dataset_name, "val", num_class)

    fig, ax = plt.subplots(1, 4, figsize=(10, 3))

    plt.axis('off')

    for i in range(4):
        inde = random.randint(0, len(train_dataset)-1)
        x_t,y_t,c_t = train_dataset[inde]
        ax[i].imshow(x_t.permute(1, 2, 0).cpu().numpy())

        ax[i].set_title(f"Task label: {y_t.numpy()}")

        ax[i].axis("off")


    plt.show()

    # x_t,y_t = val_dataset[0]
    # print(y_t)
    # img = np.transpose(np.squeeze(x_t.numpy()),(1,2,0))
    # plt.imshow(img)
    # plt.show()