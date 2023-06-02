from model.pytorchtools import EarlyStopping

import os
import wandb
import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms


def train_model(model:torch.nn.Module, loss_function, optimizer, device, epoch_num, epoch, train_datasetloader:data_utils.DataLoader, prefetcher = None):
    model.train()
    model.to(device=device)

    sum_loss = 0
    step_num = len(train_datasetloader)

    # num = 0
    with tqdm.tqdm(total= step_num) as tbar:
        if prefetcher is None:
            for data, target in train_datasetloader:
                # if epoch==0 and num==0:
                #     images = wandb.Image(
                #         data[0].squeeze(0),
                #         caption="Input sample"
                #     )
                #     wandb.log({"Input sample":images})

                #     num+=1
                data, target = data.to(device), target.to(device)

                # print("input and output", data.shape, target.shape)

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
        else:
            data, target = prefetcher.next()
            while data is not None:
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

                data, target = prefetcher.next()

    
    ave_loss = sum_loss / step_num
    return ave_loss


def val_model(model:torch.nn.Module, device, loss_function, val_datasetloader, prefetcher = None):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(val_datasetloader.dataset)
    multi_target_num = 1
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_datasetloader)) as pbar:
            if prefetcher is None:
                for data, target in val_datasetloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = loss_function(output, target)
                    if output.shape == target.shape:
                        pred = output
                        multi_target_num = target.shape[-1]
                        correct += torch.sum(pred.ge(0.5) == target)
                    else:
                        _, pred = torch.max(output.data, 1)
                        correct += torch.sum(pred == target)
                    print_loss = loss.data.item()
                    test_loss += print_loss
                    pbar.update(1)
            else:
                data, target = prefetcher.next()
                while data is not None:
                    output = model(data)
                    loss = loss_function(output, target)
                    if output.shape == target.shape:
                        pred = output
                        multi_target_num = target.shape[-1]
                        correct += torch.sum(pred.ge(0.5) == target)
                    else:
                        _, pred = torch.max(output.data, 1)
                        correct += torch.sum(pred == target)
                    print_loss = loss.data.item()
                    test_loss += print_loss
                    pbar.update(1)
                    data, target = prefetcher.next()


        correct = correct.data.item()
        acc = correct / total_num / multi_target_num
        avgloss = test_loss / len(val_datasetloader)
        # print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avgloss, correct, total_num, 100 * acc))
    
    return avgloss, correct, acc

def load_dataset(data_folder, input_size, batch_size):

    train_transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
            transforms.ToTensor()])

    val_transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_folder,'train'), transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_folder,'val'), transform=val_transform)

    print(f"{train_dataset.classes}, {train_dataset.class_to_idx}")
    print(f"{val_dataset.classes}, {val_dataset.class_to_idx}")

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

    return train_dataloader, val_dataloader


def load_model(model_name, model_parameter_path=None):
    if model_name=='vgg':
        model = torchvision.models.vgg16_bn(pretrained=True) 
        model.classifier[6] = nn.Linear(model.classifier[6].in_features,7)
    elif model_name=='resnet':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,7)
    elif model_name=='mobilenet':
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,7)
    elif model_name=="densenet":
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features,7)
    else:
        raise Exception(f"Can not find model {model_name}")

    if model_parameter_path is not None:
        model.load_state_dict(torch.load(model_parameter_path))

    print("Model Structure", model)
    return model


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

    # models = ['vgg','resnet', 'mobilenet']
    models = ['densenet']

    sample_nums = [
        36656/3504,36656/4517,
        36656/1590,
        36656/839,36656/8815,
        36656/3373,36656/1909]

    for model_name in models:
        print("Start {} model training".format(model_name))

        model = load_model(model_name)
        
        model.to(device)

        train_dataloader, val_dataloader = load_dataset(dataset_folder, input_size, batch_size)

        model_save_path = f"model/logs/{model_name}.pt"
        early_stopping = EarlyStopping(patience=20, verbose=True, path=model_save_path)

        optimizer = optim.AdamW(model.parameters(), learn_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        loss_function = nn.CrossEntropyLoss(weight=torch.Tensor(sample_nums).to(device))

        best_acc = 0
        
        for epoch in range(NUM_EPOCHES):
            train_loss = train_model(model, loss_function, optimizer, device, NUM_EPOCHES, epoch, train_dataloader)
            avgloss, correct, acc = val_model(model, device, loss_function, val_dataloader)
            scheduler.step()

            early_stopping(avgloss, acc, model, train_loss)
            log = {f'training loss {model_name}': train_loss, 
                f'val loss {model_name}': avgloss, 
                f'val acc {model_name}': acc, 
                'epoch':epoch}
            wandb.log(log)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"{model_name}_best.pt")

            if early_stopping.early_stop:
                print("Early stopping")
                break