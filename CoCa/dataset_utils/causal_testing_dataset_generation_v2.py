import torchvision
import os
import random
import numpy as np
import tqdm
from matplotlib import pyplot as plt

def random_set_red_color(pics, concept_red):

    img = pics[concept_red+2].copy()
    img[:,:,1] = 0
    img[:,:,2] = 0

    pics[concept_red+2]=img

    return pics


def digits2data(dataset, inds, dataset_type):
    pics = []; digits = []

    for ii in range(4):
        pics.append(np.asarray(dataset[inds[ii]][0].convert('RGB')))
        digits.append(dataset[inds[ii]][1])

    concept_pointer = min(digits[0],digits[1]) % 2

    if dataset_type == "collider":
        concept_red = random.randint(0,1)
    elif dataset_type == "chain":
        if digits[0] <= digits[1]:
            concept_red = random.randint(0,1)
        else:
            return None, None, None
    elif dataset_type == "fork":
        concept_red = concept_pointer

    pics = random_set_red_color(pics, concept_red)
    
    ab = np.hstack((pics[0],pics[1]))
    cd = np.hstack((pics[2],pics[3]))
    abcd = np.vstack((ab,cd))
    # plt.imshow(abcd)
    # plt.show()
    # abcd = np.expand_dims(abcd, axis = 0)

    y = digits[2+concept_pointer]

    digits.append(y)

    return abcd, y, [digits[0],digits[1],digits[2],digits[3],concept_pointer,concept_red,y]


def shift_dataset(num_sample, dataset, dataset_type):
    data_x = []; data_y = []; data_concept = []

    max_num = len(dataset)
    with tqdm.tqdm(total=num_sample) as tbar:
        while True:
            inds = [random.randint(0,max_num-1) for ii in range(4)]
            # concept a^{TL}, a^{TR}, a^{BL}, a^{BR}, a^{pointer}, a^{red}, y
            x,y,concepts = digits2data(dataset, inds, dataset_type)
            if x is None:
                continue

            data_x.append(x); data_y.append(y), data_concept.append(concepts)

            if len(data_x) == num_sample:
                break
            tbar.update()

    return data_x, data_y, data_concept

def generate_causal_pvr_minist_v2(dataset_path):
    # load MNIST dataset
    train_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    val_mnist_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True)

    # build dataset for collider structure
    random_collider_path = os.path.join(dataset_path,'collider')
    if not os.path.exists(random_collider_path):
        os.mkdir(random_collider_path)

    train_x, train_y, train_concept = shift_dataset(10000, train_mnist_dataset, 'collider')
    test_x, test_y, test_concept = shift_dataset(1000, val_mnist_dataset, 'collider')

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_collider_path,"train_image_casaul_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_collider_path,"train_label_casaul_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_collider_path,"train_concept_casaul_pvr.npy") ,np.array(train_concept))
    np.save(os.path.join(random_collider_path,"test_image_casaul_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_collider_path,"test_label_casaul_pvr.npy") ,np.array(test_y))
    np.save(os.path.join(random_collider_path,"test_concept_casaul_pvr.npy") ,np.array(test_concept))

    # build dataset for chain structure
    random_chain_path = os.path.join(dataset_path,'chain')
    if not os.path.exists(random_chain_path):
        os.mkdir(random_chain_path)

    train_x, train_y, train_concept = shift_dataset(10000, train_mnist_dataset, 'chain')
    test_x, test_y, test_concept = shift_dataset(1000, val_mnist_dataset, 'chain')

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_chain_path,"train_image_casaul_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_chain_path,"train_label_casaul_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_chain_path,"train_concept_casaul_pvr.npy") ,np.array(train_concept))
    np.save(os.path.join(random_chain_path,"test_image_casaul_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_chain_path,"test_label_casaul_pvr.npy") ,np.array(test_y))
    np.save(os.path.join(random_chain_path,"test_concept_casaul_pvr.npy") ,np.array(test_concept))

    # build dataset for fork structure
    random_fork_path = os.path.join(dataset_path,'fork')
    if not os.path.exists(random_fork_path):
        os.mkdir(random_fork_path)

    train_x, train_y, train_concept = shift_dataset(10000, train_mnist_dataset, 'fork')
    test_x, test_y, test_concept = shift_dataset(1000, val_mnist_dataset, 'fork')

    print(type(train_x),type(train_y),type(train_x[10]))
    print(np.array(train_x).shape, np.array(train_y).shape)

    np.save(os.path.join(random_fork_path,"train_image_casaul_pvr.npy") ,np.array(train_x))
    np.save(os.path.join(random_fork_path,"train_label_casaul_pvr.npy") ,np.array(train_y))
    np.save(os.path.join(random_fork_path,"train_concept_casaul_pvr.npy") ,np.array(train_concept))
    np.save(os.path.join(random_fork_path,"test_image_casaul_pvr.npy") ,np.array(test_x))
    np.save(os.path.join(random_fork_path,"test_label_casaul_pvr.npy") ,np.array(test_y))
    np.save(os.path.join(random_fork_path,"test_concept_casaul_pvr.npy") ,np.array(test_concept))



if  __name__ == "__main__":

    generate_causal_pvr_minist_v2("F:\\causal_pvr_v2")
