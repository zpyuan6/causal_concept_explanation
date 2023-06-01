import os
import wandb
import tqdm
from PIL import Image

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
        self.features.append(input[0].cpu().clone())

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_parameter_path = "model\\logs\\resnet.pt"
    model_name = "resnet"

    model = load_model("model\\logs\\resnet.pt", "resnet")
    model.to(device)
    print(model)

    train_folder = "F:\\Broden\\opensurfaces\\train"
    val_folder = "F:\\Broden\\opensurfaces\\train"

    save_path = "F:\\Broden\\feature_maps"

    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

    layers_name_vgg = ["maxpool","layer1","layer2","layer3","layer4","fc"]
    layers_name_resnet = ["maxpool","layer1","layer2","layer3","layer4","fc"]

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
                np_path = os.path.join(save_path,f"{model_name}_{layers_name_resnet[i]}", root.split("\\")[-1])
                if not os.path.exists(np_path):
                    os.makedirs(np_path) 
                torch.save(feature_map, os.path.join(np_path,file.split(".")[0]+".pt")) 

            hooks.clean_features()


    for root, folders, files in os.walk(val_folder):
        for file in files:
            img = Image.open(os.path.join(root,file))
            input_img = transform(img)
            input_img = input_img.to(device)
            input_img = torch.unsqueeze(input_img, 0)
            output = model(input_img)

            print(torch.argmax(output,dim=1))

    

