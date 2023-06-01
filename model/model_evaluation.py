import os
import wandb
import tqdm
from PIL import Image

from model.model_training import load_model

import torchmetrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

def load_dataset(data_folder):
    val_transform=transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()])

    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_folder,'val'), transform=val_transform)

    print(f"{val_dataset.__len__()} {val_dataset.classes}, {val_dataset.class_to_idx}")

    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=6, pin_memory = True, prefetch_factor=8)

    return val_dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_parameter_path = "model\\logs\\resnet.pt"
    model_name = "resnet"

    model = load_model("resnet", "model\\logs\\resnet.pt", )
    # model = load_model("mobilenet", "model\logs\mobilenet.pt")
    model.to(device)

    val_folder = "F:\\Broden\\opensurfaces"

    val_dataloader = load_dataset(val_folder)

    
    test_acc = torchmetrics.Accuracy().to(device)
    test_recall = torchmetrics.Recall(average='none', num_classes=7).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=7).to(device)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=7).to(device)

    total_num = len(val_dataloader.dataset)

    TP = torch.zeros(7)
    FP = torch.zeros(7)
    TN = torch.zeros(7)
    FN = torch.zeros(7)

    with torch.no_grad():
        with tqdm.tqdm(total = len(val_dataloader)) as pbar:
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                test_acc(torch.argmax(output, 1), target)
                test_auc.update(output, target)
                test_recall(torch.argmax(output, 1), target)
                test_precision(torch.argmax(output, 1), target)

                pbar.update(1)

    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    print(f"torch metrics acc: {(100 * total_acc):>0.1f}%\n")
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("auc:", total_auc.item())

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()