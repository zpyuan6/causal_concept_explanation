from model.pytorchtools import EarlyStopping
from model.model_training import val_model,train_model
from data.concept_dataset import ConceptDataset

import os
import wandb
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

def append_auxiliary_linear_layers(model, model_name, num_concepts:int):
    model

def load_model(model_parameter_path, model_name):
    model = torchvision.models.vgg11(pretrained=True) 
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,20)
    if model_name=='resnet':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,20)
    elif model_name=='mobilenet':
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,20)
    print(model)

    model.load_state_dict(torch.load(model_parameter_path))

    for param in model.named_parameters():
        param[1].requires_grad = False

    return model

def load_dataset(concept_type, batch_size):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    train_dataset =  ConceptDataset(train_or_val="train", concept_type=concept_type)
    val_dataset =  ConceptDataset(train_or_val="val", concept_type=concept_type)

    print(f"training samples: {train_dataset.__len__()}, val samples: {val_dataset.__len__()} for concept {concept_type}")

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

    return train_dataloader, val_dataloader


def train_and_val_auxiliary_layer(batch_size, learn_rate, num_epoches, concept_type, model:torch.nn.Module):
    train_dataloader, val_dataloader = load_dataset(concept_type, batch_size)

    append_auxiliary_linear_layer(model)



if __name__ == "__main__":

    batch_size = 8
    learn_rate=0.001
    num_epoches=300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_parameter_path = "model\\logs\\resnet.pt"
    concept_type = "color"

    model = load_model(model_parameter_path, "resnet")

    train_and_val_auxiliary_layer(batch_size, learn_rate, num_epoches, concept_type, model)
