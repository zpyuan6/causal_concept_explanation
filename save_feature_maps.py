import os
import wandb
import tqdm
from PIL import Image

from model.model_training import load_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

class ModelHook():
    def __init__(self) -> None:
        self.features = []
        self.hooks = []

    def hook(self, module, input, output):
        self.features.append(input[0].cpu().clone().detach())

    def register_hook(self,model:nn.Module,layer_name:str):
        hook = model._modules[layer_name].register_forward_hook(self.hook)
        self.hooks.append(hook)

    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
        self.features = []

    def clean_features(self):
        del self.features
        self.features = []

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "resnet"

    model_parameter_path = f"model\\logs\\{model_name}_best.pt"
    model = load_model(model_name,model_parameter_path)
    model.to(device)
    print(model)

    train_folder = "F:\\Broden\\opensurfaces\\train"
    val_folder = "F:\\Broden\\opensurfaces\\val"

    save_path = "F:\\Broden\\concept_model\\feature_maps"

    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

    layers_name_vgg = ["maxpool","layer1","layer2","layer3","layer4","fc"]
    layers_name_resnet = ["maxpool","layer1","layer2","layer3","layer4","fc"]
    layers_name_mobilenet = ["maxpool","layer1","layer2","layer3","layer4","fc"]

    hooks = ModelHook()
    for name in layers_name_resnet:
        hooks.register_hook(model, name)

    for root, folders, files in os.walk(train_folder):
        print(root, "is processing.")
        for file in files:
            img = Image.open(os.path.join(root,file)).convert("RGB")
            input_img = transform(img)
            input_img = input_img.to(device)
            input_img = torch.unsqueeze(input_img, 0)
            output = model(input_img)

            # print(torch.argmax(output,dim=1))

            for i, feature_map in enumerate(hooks.features):
                np_path = os.path.join(save_path,f"{model_name}_{layers_name_resnet[i]}","train" ,root.split("\\")[-1])
                if not os.path.exists(np_path):
                    os.makedirs(np_path) 
                torch.save(feature_map, os.path.join(np_path,file.split(".")[0]+".pt")) 

            hooks.clean_features()


    for root, folders, files in os.walk(val_folder):
        print(root, "is processing.")
        for file in files:
            img = Image.open(os.path.join(root,file)).convert("RGB")
            input_img = transform(img)
            input_img = input_img.to(device)
            input_img = torch.unsqueeze(input_img, 0)
            output = model(input_img)

            # print(torch.argmax(output,dim=1))

            for i, feature_map in enumerate(hooks.features):
                np_path = os.path.join(save_path,f"{model_name}_{layers_name_resnet[i]}", "val", root.split("\\")[-1])
                if not os.path.exists(np_path):
                    os.makedirs(np_path) 
                torch.save(feature_map, os.path.join(np_path,file.split(".")[0]+".pt")) 

            hooks.clean_features()

    
