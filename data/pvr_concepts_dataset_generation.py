import torch
import torchvision
import torch.utils.data as data

import os
import random
import numpy as np
import pickle
import PIL
import pickle

class PVRConceptDataset(data.Dataset):
    def __init__(self, dataset_path) -> None:
        super().__init__()
        file = open(dataset_path,'rb')
        x, y = pickle.load(file)

        self.x = torch.from_numpy(np.array(x))/255

        if self.x.shape[1] != 3:
            self.x = self.x.repeat(1,3,1,1)

        print(f"Load from {dataset_path}",self.x.shape)
        # print(self.x.shape, torch.unsqueeze(torch.from_numpy(np.array(y)),1).dtype)
        # self.y = torch.nn.functional.one_hot(torch.unsqueeze(torch.from_numpy(np.array(y)),1).to(torch.int64) , num_class)
        self.y = torch.from_numpy(np.array(y)).to(torch.int64)


    def __getitem__(self, index):
        return  self.x[index], self.y[index]


    def __len__(self):
        return len(self.x)

class FeatureMapsDataset(data.Dataset):
    def __init__(self, x:torch.tensor, y:torch.tensor) -> None:
        self.x = x
        self.y = y

        # print(f"feature maps dataset {self.x.shape} {self.y.shape}")

    def __getitem__(self, index):
        return  self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


CONCEPT_LIST = ['TOP_LEFT','TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'IS_ZERO', 'IS_ONE', 'IS_TWO', 'IS_THREE', 'IS_FOUR','IS_FIVE','IS_SIX', 'IS_SEVEN', 'IS_EIGHT','IS_NINE']

def get_concept_label(concept_name, labels):
    index = CONCEPT_LIST.index(concept_name)
    
    if index == 0:
        return labels[0]
    elif index == 1:
        return labels[1]

    elif index == 2:
        return labels[2]

    elif index == 3:
        return labels[3]
        
    elif index == 4:
        if 0 in labels:
            return True
        else:
            return False

    elif index == 5:
        if 1 in labels:
            return True
        else:
            return False
        
    elif index == 6:
        if 2 in labels:
            return True
        else:
            return False

    elif index == 7:
        if 3 in labels:
            return True
        else:
            return False

    elif index == 8:
        if 4 in labels:
            return True
        else:
            return False

    elif index == 9:
        if 5 in labels:
            return True
        else:
            return False
        
    elif index == 10:
        if 6 in labels:
            return True
        else:
            return False

    elif index == 11:
        if 7 in labels:
            return True
        else:
            return False

    elif index == 12:
        if 8 in labels:
            return True
        else:
            return False

    elif index == 13:
        if 9 in labels:
            return True
        else:
            return False

    raise Exception(f"can not found concept {concept_name}")

def generate_PVRConceptDataset(dataset_name, dataset_path, target_path, number_of_samples=[10], is_training=False):

    mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=False)
    mnist_len = len(mnist_dataset)
    cifar_dataset = None
    cifar_len = 0
    if dataset_name =='cifar':
        cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False)
        cifar_len = len(cifar_dataset)

    if not os.path.exists(target_path):
        os.mkdir(target_path)

    if dataset_name == 'mnist':
        for concept_name in CONCEPT_LIST:
            for num in number_of_samples:
                dataset_x, dataset_y = [],[]
                y_map = {}
                num_samples = num * 10 if CONCEPT_LIST.index(concept_name)<4 else num * 2
                while len(dataset_x)<num_samples:
                    labels, imgs = [],[]
                    for ii in range(4):
                        idx = random.randint(0, mnist_len-1)
                        imgs.append(mnist_dataset[idx][0])
                        labels.append(mnist_dataset[idx][1])

                    ab = np.hstack((imgs[0], imgs[1]))
                    cd = np.hstack((imgs[2], imgs[3]))
                    abcd = np.vstack((ab,cd))
                    abcd = np.expand_dims(abcd, axis = 0)

                    if (not get_concept_label(concept_name,labels) in y_map):
                        y_map[get_concept_label(concept_name,labels)] = 1
                        dataset_x.append(abcd)
                        dataset_y.append(get_concept_label(concept_name,labels))

                    if (y_map[get_concept_label(concept_name,labels)]<num):
                        y_map[get_concept_label(concept_name,labels)] = y_map[get_concept_label(concept_name,labels)]+1
                        dataset_x.append(abcd)
                        dataset_y.append(get_concept_label(concept_name,labels))
                
                concept_dataset_path = os.path.join(target_path,f"{dataset_name}_{concept_name}_{num}.txt") if is_training else os.path.join(target_path,f"val_{dataset_name}_{concept_name}_{num}.txt")
                print(concept_dataset_path)
                with open(concept_dataset_path, "wb") as fp:  
                    pickle.dump((dataset_x, dataset_y), fp)
        return


    elif dataset_name == 'cifar':
        for concept_name in CONCEPT_LIST:
            for num in number_of_samples:
                dataset_x, dataset_y = [],[]
                y_map = {}
                num_samples = num * 10 if CONCEPT_LIST.index(concept_name)<4 else num * 2
                while len(dataset_x)<num_samples:
                    labels, imgs = [],[]
                    idx = random.randint(0, mnist_len-1)
                    imgs.append(mnist_dataset[idx][0].convert('RGB'))
                    labels.append(mnist_dataset[idx][1])
                    for ii in range(1,4):
                        idx = random.randint(0, cifar_len-1)
                        imgs.append(cifar_dataset[idx][0])
                        labels.append(cifar_dataset[idx][1])

                    abcd = PIL.Image.new(imgs[0].mode, (64,64),(0,0,0))
                    abcd.paste(imgs[0], (2,2))
                    abcd.paste(imgs[1], (32,2))
                    abcd.paste(imgs[2], (2,32))
                    abcd.paste(imgs[3], (32,32))
                    abcd = np.array(abcd).transpose(2,0,1)

                    if (not get_concept_label(concept_name,labels) in y_map):
                        y_map[get_concept_label(concept_name,labels)] = 1
                        dataset_x.append(abcd)
                        dataset_y.append(get_concept_label(concept_name,labels))

                    if (y_map[get_concept_label(concept_name,labels)]<num):
                        y_map[get_concept_label(concept_name,labels)] = y_map[get_concept_label(concept_name,labels)]+1
                        dataset_x.append(abcd)
                        dataset_y.append(get_concept_label(concept_name,labels))
                
                concept_dataset_path = os.path.join(target_path,f"{dataset_name}_{concept_name}_{num}.txt") if is_training else os.path.join(target_path,f"val_{dataset_name}_{concept_name}_{num}.txt")
                print(concept_dataset_path)
                with open(concept_dataset_path, "wb") as fp:  
                    pickle.dump((dataset_x, dataset_y), fp)
        return
    
    raise Exception(f"Can not found dataset {dataset_name}")



if __name__ == "__main__":

    dataset_name = 'cifar'
    dataset_path = "F:\\pvr_dataset"
    target_path = os.path.join(dataset_path, f"{dataset_name}_concept") 
    generate_PVRConceptDataset(dataset_name, dataset_path, target_path, [10,20,40,80,160,320,640], False)
    generate_PVRConceptDataset(dataset_name, dataset_path, target_path, [10,20,40,80,160,320,640], True)

    dataset_name = 'mnist'
    dataset_path = "F:\\pvr_dataset"
    target_path = os.path.join(dataset_path, f"{dataset_name}_concept") 
    generate_PVRConceptDataset(dataset_name, dataset_path, target_path, [10,20,40,80,160], False)
    generate_PVRConceptDataset(dataset_name, dataset_path, target_path, [10,20,40,80,160], True)
    # generate_PVRConceptDataset(dataset_name, dataset_path, target_path, [640])
