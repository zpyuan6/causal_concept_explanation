from pytorchtools import EarlyStopping

import os
import wandb
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms


def train_model(model:torch.nn.Module, loss_function, optimizer, device, epoch_num, epoch, train_datasetloader:data_utils.DataLoader):
    model.train()
    model.to(device=device)

    sum_loss = 0
    step_num = len(train_datasetloader)

    with tqdm.tqdm(total= step_num) as tbar:
        for data, target in train_datasetloader:
            data, target = data.to(device), target.to(device)
            if data.shape[0] == 1:
                continue
            output = model(data)
            # print(output.shape, target.shape)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            
            tbar.set_description('Training Epoch: {}/{} Loss: {:.6f}'.format(epoch, epoch_num, loss.item()))
            tbar.update(1)
    
    ave_loss = sum_loss / step_num
    return ave_loss


def val_model(model:torch.nn.Module, device, loss_function, val_datasetloader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(val_datasetloader.dataset)
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_datasetloader)) as pbar:
            for data, target in val_datasetloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_function(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss
                pbar.update(1)

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(val_datasetloader)
        # print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avgloss, correct, total_num, 100 * acc))
    
    return avgloss, correct, acc

def load_dataset(data_folder, input_size, batch_size):
    transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_folder,'train'), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_folder,'val'), transform=transform)

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

    return train_dataloader, val_dataloader



if __name__ == "__main__":
    
    batch_size = 8
    dataset_folder = "F:\Broden\opensurfaces"
    input_size = [224, 224]
    learn_rate=0.001
    NUM_EPOCHES=300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(
        project="causal_concept_explanation",
    )

    models = ['vgg', 'resnet', 'mobilenet']

    for model_name in models:
        print("Start {} model training".format(model_name))

        model = torchvision.models.vgg11(pretrained=True) 
        model.classifier[6] = nn.Linear(model.classifier[6].in_features,20)
        if model_name=='resnet':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features,20)
        elif model_name=='mobilenet':
            model = torchvision.models.mobilenet_v3_small(pretrained=True)
        print(model)
        

        model.to(device)

        train_dataloader, val_dataloader = load_dataset(dataset_folder, input_size, batch_size)

        model_save_path = f"model/logs/{model_name}.pt"
        early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path)

        optimizer = optim.AdamW(model.parameters(), learn_rate)

        loss_function = nn.CrossEntropyLoss()

        for epoch in range(NUM_EPOCHES):
            train_loss = train_model(model, loss_function, optimizer, device, NUM_EPOCHES, epoch, train_dataloader)
            avgloss, correct, acc = val_model(model, device, loss_function, val_dataloader)

            early_stopping(avgloss, acc, model, train_loss)
            log = {f'training loss {model_name}': train_loss, 
                f'val loss {model_name}': avgloss, 
                f'val acc {model_name}': acc, 
                'epoch':epoch}
            wandb.log(log)
            if early_stopping.early_stop:
                print("Early stopping")
                break


        break