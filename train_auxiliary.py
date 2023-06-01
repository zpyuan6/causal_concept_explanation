from model.pytorchtools import EarlyStopping
from model.model_training import val_model,train_model
from data.concept_dataset import ConceptDataset

import os
import wandb
import tqdm
import socket
import time

from model.model_training import train_model, val_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_auxiliary_layer(input_shape, concept_num, model_parameter_path=None):

    model = nn.Sequential(
        nn.Linear(input_shape,concept_num),
        nn.Sigmoid()
    )

    if model_parameter_path is not None:
        model.load_state_dict(torch.load(model_parameter_path))

    print("Model Structure", model)
    return model
        

def load_concept_data(model_name, layer_name, concept, batch_size, device):

    train_dataset =  ConceptDataset(model_name, layer_name, device, concept, train_or_val="train")
    val_dataset =  ConceptDataset(model_name, layer_name, device, concept, train_or_val="val")

    print(f"training samples: {train_dataset.__len__()}, val samples: {val_dataset.__len__()} for concept {concept_type}, model {model_name}, layer {layer_name}")

    train_dataloader = DataLoaderX(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = False, prefetch_factor=batch_size*2)
    val_dataloader = DataLoaderX(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = False, prefetch_factor=batch_size*2)

    input_shape = train_dataset.__getitem__(0)[0].shape

    print(f"input shape is {input_shape}, {type(input_shape[0])}")

    return train_dataloader, val_dataloader, input_shape


def train_and_val_auxiliary_layer(learn_rate, num_epoches, model:torch.nn.Module, device, train_dataloader, val_dataloader, model_name, save_folder):
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), learn_rate)
    loss_function = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    early_stopping = EarlyStopping(patience=20, verbose=True, path=os.path.join(save_folder,model_name))

    best_acc=0
    for epoch in range(num_epoches):
        train_loss = train_model(model, loss_function, optimizer, device, num_epoches, epoch, train_dataloader)
        avgloss, _, acc = val_model(model, device, loss_function, val_dataloader)
        scheduler.step()

        early_stopping(avgloss, acc, model, train_loss)
        log = {f'training loss {model_name}': train_loss, 
                f'val loss {model_name}': avgloss, 
                f'val acc {model_name}': acc, 
                'epoch':epoch}
        wandb.log(log)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),  os.path.join(save_folder,model_name.split(".")[0]+"_best.pt"))

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return best_acc


if __name__ == "__main__":

    batch_size = 8
    learn_rate=0.001
    num_epoches=300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_parameter_path = "model\\logs\\resnet.pt"
    concept_type = "color"
    save_folder = "concept_models"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(
        project="causal_concept_explanation",
        name=f"{socket.gethostname()}_{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}"
    )

    # models = ['vgg','resnet', 'mobilenet']
    models = ['resnet']

    models_layers = {
        'vgg': [],
        'resnet': ["maxpool","layer1","layer2","layer3","layer4","fc"], 
        'mobilenet': []
        }

    concept_list = ["color", "material" , "part", "object" ]
    

    for model_name in models:

        for concept in concept_list:

            for layer_name in models_layers[model_name]:

                save_path = os.path.join(save_folder,f"{model_name}_{layer_name}_{concept}.pt")

                train_dataloader, val_dataloader, input_shape = load_concept_data(model_name, layer_name, concept, batch_size, device)

                model = build_auxiliary_layer(input_shape[0], concept_num = len(train_dataloader.dataset.concept_indexs)) 

                train_and_val_auxiliary_layer(learn_rate, num_epoches, model, device, train_dataloader, val_dataloader, model_name=f"{concept}_{layer_name}_{model_name}.pt", save_folder=save_folder)

            
