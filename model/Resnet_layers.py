from model.model_training import load_model
import torch.nn as nn

LAYERS = ["maxpool","layer1","layer2","layer3","layer4","fc"]

def get_model(start_index):
    model = load_model("resnet", "model\\logs\\resnet_best.pt")

    model = nn.Sequential()

    start_loading = False
    add_layer = ""
    for name, module in model.named_modules():
        if name == LAYERS(start_index):
            start_loading=True
            
        if start_index<len(LAYERS)-1 and name == LAYERS(start_index+1):
            break

        if start_loading and (not add_layer in name):
            model.add_module(module)
            print(name, module.parameters())
            add_layer=name

    print(model)
    print(model[-1].parameters())

    return model

