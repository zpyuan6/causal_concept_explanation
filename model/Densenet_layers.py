from model.model_training import load_model
import torch.nn as nn

LAYERS = ["features.denseblock1", "features.denseblock2","features.denseblock3","features.denseblock4","classifier"]

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
            add_layer=name

    return model