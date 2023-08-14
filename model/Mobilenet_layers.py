from model.model_training import load_model
import torch.nn as nn

LAYERS = ["features.3","features.6","features.9","features.12","classifier"]


def get_model(start_index):
    model = load_model("mobilenet", "model\\logs\\mobilenet_best.pt")

    layers_model = nn.Sequential()

    start_loading = False
    add_layer = " "
    index=0
    for name, module in model.named_modules():
        if name == LAYERS[start_index]:
            start_loading=True
            
        if start_index<len(LAYERS)-1 and name == LAYERS[start_index+1]:
            start_loading=False
            break

        if start_loading:
            if add_layer in name:
                continue
            print("add---------", name, module)
            layers_model.add_module(f"feature{index}", module)
            index+=1
            add_layer=name

    return layers_model