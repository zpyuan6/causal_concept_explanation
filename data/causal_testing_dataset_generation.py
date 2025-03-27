import torchvision
import os
import random
import numpy as np
import pickle
import tqdm

def digits2data(dataset, inds, dataset_type):
    pics = []; digits = []
    for ii in range(4):
        pics.append(np.asarray(dataset[inds[ii] ][0]))
        digits.append(dataset[inds[ii]][1])
    
    ab = np.hstack((pics[0],pics[1]))
    cd = np.hstack((pics[2],pics[3]))
    abcd = np.vstack((ab,cd))
    abcd = np.expand_dims(abcd, axis = 0)
    
    pointer = max(0,digits[0]-1)//3 + 1

    if dataset_type == "collider":
        if pointer == 1:
            y = int((digits[1]+digits[2])/2)
        elif pointer == 2:
            y = int((digits[2]+digits[3])/2)
        elif pointer == 3:
            y = int((digits[3]+digits[1])/2)
    else:
        y = digits[pointer]

    digits.append(y)

    return abcd, digits[pointer], digits

def sample_id(dataset, criterion, dataset_type):
    max_num = len(dataset)
    ids = []
    index_point = 0
    for ii in range(4):
        idx = random.randint(0,max_num-1)
        crit = criterion[ii]
        if dataset_type == "fork" and ii>0:
            pointer = max(0,index_point-1)//3 + 1
            if pointer == 1 and ii==2:
                crit = [dataset[ids[1]][1]+1 if (dataset[ids[1]][1]+1)<10 else 0]
            elif pointer == 2 and ii==3:
                crit = [dataset[ids[2]][1]+1 if (dataset[ids[2]][1]+1)<10 else 0]
            elif pointer == 3 and ii==3:
                crit = [dataset[ids[1]][1]-1 if (dataset[ids[1]][1]-1)>=0 else 9]
            else:
                crit = criterion[ii]
            # print(pointer,ii, crit)
        while(not(dataset[idx][1] in crit)):
            idx = random.randint(0,max_num-1)
        ids.append(idx)
        if ii == 0:
            index_point = dataset[idx][1]

    return ids

def shift_dataset(num_sample, dataset, dataset_type, c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]]):
    data_x = []; data_y = []; data_z = []

    with tqdm.tqdm(total=num_sample) as tbar:
        for ii in range(num_sample):
            inds = sample_id(dataset, c, dataset_type)
            x,y,z = digits2data(dataset, inds, dataset_type)
            data_x.append(x); data_y.append(y), data_z.append(z)
            tbar.update()

    return data_x, data_y

def generate_causal_pvr_minist(dataset_path):

    # load MNIST dataset
    train_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    val_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True)

    # build dataset for chain structure
    random_chain_path = os.path.join(dataset_path,'random_chain')
    if not os.path.exists(random_chain_path):
        os.mkdir(random_chain_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'chain', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'chain', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_chain_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_chain_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_chain_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_chain_path,"test_label_mnist_pvr.npy") ,np.array(test_y))

    limited_random_chain_path = os.path.join(dataset_path,'limited_random_chain')
    if not os.path.exists(limited_random_chain_path):
        os.mkdir(limited_random_chain_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'chain', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'chain', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(limited_random_chain_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(limited_random_chain_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(limited_random_chain_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(limited_random_chain_path,"test_label_mnist_pvr.npy") ,np.array(test_y))


    random_fork_path = os.path.join(dataset_path,'random_fork')
    if not os.path.exists(random_fork_path):
        os.mkdir(random_fork_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'fork', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'fork', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_fork_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_fork_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_fork_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_fork_path,"test_label_mnist_pvr.npy") ,np.array(test_y))


    limited_random_fork_path = os.path.join(dataset_path,'limited_random_fork')
    if not os.path.exists(limited_random_fork_path):
        os.mkdir(limited_random_fork_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'fork', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'fork', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(limited_random_fork_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(limited_random_fork_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(limited_random_fork_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(limited_random_fork_path,"test_label_mnist_pvr.npy") ,np.array(test_y))

    random_collider_path = os.path.join(dataset_path,'random_collider')
    if not os.path.exists(random_collider_path):
        os.mkdir(random_collider_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'collider', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'collider', c=[[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_collider_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_collider_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_collider_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_collider_path,"test_label_mnist_pvr.npy") ,np.array(test_y))

    limited_random_collider_path = os.path.join(dataset_path,'limited_random_collider')
    if not os.path.exists(limited_random_collider_path):
        os.mkdir(limited_random_collider_path)

    train_x, train_y = shift_dataset(10000, train_mnist_dataset, 'collider', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])
    test_x, test_y = shift_dataset(1000, val_mnist_dataset, 'collider', c=[[1,2,3,4,5,6,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0],[1,2,3,4,5,6,7,8,9,0]])

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(limited_random_collider_path,"train_image_mnist_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(limited_random_collider_path,"train_label_mnist_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(limited_random_collider_path,"test_image_mnist_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(limited_random_collider_path,"test_label_mnist_pvr.npy") ,np.array(test_y))

if  __name__ == "__main__":

    generate_causal_pvr_minist("F:\\pvr_dataset\\causal_validation_pvr")
