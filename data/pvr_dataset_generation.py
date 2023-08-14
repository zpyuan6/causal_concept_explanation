
import torchvision
from torch.utils.data import ConcatDataset
import numpy as np
import random
import PIL
import matplotlib.pyplot as plt
import os
import pickle

def get_mnist_and_cifar10(dataset_path):
    dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True)
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True)
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True)


def digits2data(dataset, inds):
    pics = []; digits = []
    for ii in range(4):
        pics.append(np.asarray(dataset[inds[ii] ][0]))
        digits.append(dataset[inds[ii] ][1])
    
    ab = np.hstack((pics[0],pics[1]))
    cd = np.hstack((pics[2],pics[3]))
    abcd = np.vstack((ab,cd))
    abcd = np.expand_dims(abcd, axis = 0)
    
    pointer = max(0,digits[0]-1)//3 + 1

    return abcd, digits[pointer], np.expand_dims(pics[pointer], axis = 0)

def sample_id(dataset, criterion):
    max_num = len(dataset)
    ids = [random.randint(0,max_num-1)]
    for ii in range(3):
        idx = random.randint(0,max_num-1)
        while(not(dataset[idx][1] in criterion[ii])):
            idx = random.randint(0,max_num-1)
        ids.append(idx)
    return ids

def shift_dataset(num_sample, dataset, is_train, AE = False):
    data_x = []; data_y = []; data_z = []
    if(is_train):
        # c = [[4,5,6,7,8,9,0], [1,2,3,7,8,9,0], [1,2,3,4,5,6]]
        c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]]
    else:
        c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]]
        # c = [[1,2,3], [4,5,6], [7,8,9,0]]
    for ii in range(num_sample):
        inds = sample_id(dataset, c)
        x,y,z = digits2data(dataset, inds)
        data_x.append(x); data_y.append(y), data_z.append(z)
    if(AE):
        return data_x, data_z
    else:
        return data_x, data_y

def generate_pvr_minist(dataset_path):
    train_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=False)
    val_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=False)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, is_train = True)
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, is_train = False)

    print(type(train_x),type(train_y),type(train_x[10]))

    print(np.array(train_x).shape, np.array(train_y).shape)

    dataset_path = os.path.join(dataset_path,"mnist_pvr")
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    with open(os.path.join(dataset_path,"train_mnist_pvr.txt"), "wb") as fp:   
        pickle.dump((train_x, train_y), fp)
    with open(os.path.join(dataset_path,"val_mnist_pvr.txt"), "wb") as fp:  
        pickle.dump((test_x, test_y), fp)

    np.save(os.path.join(dataset_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(dataset_path,"train_label_mnist_pvr.npy") ,np.array(train_y))

    np.save(os.path.join(dataset_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(dataset_path,"test_label_mnist_pvr.npy") ,np.array(test_y))


    # idx = random.randint(0,1000)
    # image = PIL.Image.fromarray(np.squeeze(train_x[idx]) )
    # plt.imshow(image)
    # plt.show()
    # print(train_y[idx])

    # train_cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False)
    # val_cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False)


def generate_pvr_cifar(dataset_path):
    train_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=False)
    val_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=False)

    train_cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=False)
    val_cifar_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False)

    print(len(train_cifar_dataset),len(val_cifar_dataset))


    dataset_path = os.path.join(dataset_path,"cifar_pvr")
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    train_x, train_y = [],[]

    for i in range(10000):
        max_num = len(train_mnist_dataset)
        p_index = random.randint(0,max_num-1)
        pics = []
        pics.append(train_mnist_dataset[p_index][0].convert('RGB'))
        pvr = int((train_mnist_dataset[p_index][1]-1)/3) + 1
        print(pvr)

        max_num = len(train_cifar_dataset)
        for i in range(1,4):
            i_index = random.randint(0,max_num-1)
            pics.append(train_cifar_dataset[i_index][0])
            if pvr == i:
                train_y.append(train_cifar_dataset[i_index][1])

        abcd = PIL.Image.new(pics[0].mode, (64,64),(0,0,0))
        abcd.paste(pics[0], (2,2))
        abcd.paste(pics[1], (32,2))
        abcd.paste(pics[2], (2,32))
        abcd.paste(pics[3], (32,32))
        
        abcd = np.array(abcd).transpose(2,0,1)
        train_x.append(abcd)

    np.save(os.path.join(dataset_path,"train_image_cifar_pvr.npy"), np.array(train_x))
    np.save(os.path.join(dataset_path,"train_label_cifar_pvr.npy"), np.array(train_y))

    with open(os.path.join(dataset_path,"train_cifar_pvr.txt"), "wb") as fp:   
        pickle.dump((train_x, train_y), fp)
    
            
    test_x, test_y = [],[]

    for i in range(1000):
        max_num = len(val_mnist_dataset)
        p_index = random.randint(0,max_num-1)
        pics = []
        pics.append(val_mnist_dataset[p_index][0].convert('RGB'))
        pvr = int((val_mnist_dataset[p_index][1]-1)/3) + 1
        print(pvr)

        max_num = len(val_cifar_dataset)
        for i in range(1,4):
            i_index = random.randint(0,max_num-1)
            pics.append(val_cifar_dataset[i_index][0])
            if pvr == i:
                test_y.append(val_cifar_dataset[i_index][1])

        abcd = PIL.Image.new(pics[0].mode, (64,64),(0,0,0))
        abcd.paste(pics[0], (2,2))
        abcd.paste(pics[1], (32,2))
        abcd.paste(pics[2], (2,32))
        abcd.paste(pics[3], (32,32))
        
        abcd = np.array(abcd).transpose(2,0,1)
        test_x.append(abcd)

    np.save(os.path.join(dataset_path,"val_image_cifar_pvr.npy"), np.array(test_x))
    np.save(os.path.join(dataset_path,"val_label_cifar_pvr.npy"), np.array(test_y))

    with open(os.path.join(dataset_path,"val_cifar_pvr.txt"), "wb") as fp:  
        pickle.dump((test_x, test_y), fp)

if __name__ == "__main__":
    dataset_path = "F:\\pvr_dataset"
    # get_mnist_and_cifar10(dataset_path)
    # generate_pvr_minist(dataset_path)
    generate_pvr_cifar(dataset_path)