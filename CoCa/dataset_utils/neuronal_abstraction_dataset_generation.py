
import os
from tqdm import tqdm
from PIL import Image
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.model_training import load_model
from data.PVRDataset import CausalPVRDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy

import torch.nn.functional as F

class ModelHook():
    def __init__(self) -> None:
        self.features = []
        self.hooks = []

    def hook(self, module, input, output):
        # print(input)
        # self.features.append(input[0].cpu().clone().detach())
        if len(output.shape) == 4:
            neural_feature = output.mean(dim=[2,3])
        elif len(output.shape) == 2:
            neural_feature = output
        else:
            raise ValueError("Output shape is not 2 or 4")
        
        self.features.append(neural_feature.cpu().clone().detach())
        # print(output.shape)

    def register_hook(self,model:nn.Module,layer_name:str, module:nn.Module=None):
        if module ==None:
            hook = model._modules[layer_name].register_forward_hook(self.hook)
        else:
            hook = module.register_forward_hook(self.hook)
        self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.features = []

    def clean_features(self):
        del self.features
        self.features = []

def generate_neuronal_abstraction_dataset(dataset_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ['resnet', 'mobilenet']

    data_types = ['test', 'train']

    for model_name in models:
        model_parameter_path = f"{dataset_folder}\\{model_name}_best.pt"
        model = load_model(model_name,model_parameter_path, num_class=10)
        model.to(device)
        model.eval()

        hooks = ModelHook()

        if model_name=="resnet":
            layers_names = ["conv1","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
            for name in layers_names:
                hooks.register_hook(model, name)
        elif model_name == "mobilenet":
            layers_names = ["features.0","features.1","features.2","features.3","features.4","features.5","features.6","features.7","features.8","features.9","features.10","features.11","features.12","avgpool","classifier"]
            for name,module in model.named_modules():
                if name in layers_names:
                    hooks.register_hook(model, name,module)

        for dt in data_types:
            dataset = CausalPVRDataset(dataset_folder, dt)

            dataloader = DataLoader(dataset, shuffle=False, batch_size=16, num_workers=6, pin_memory = True, prefetch_factor=16*2)

            neuronal_abstractions = []
            concepts = []

            with tqdm(total= len(dataloader)) as tbar:
                for data, target, concept in dataloader:
                    data = data.to(device)
                    concepts.append(concept.numpy())

                    output = model(data)

                    max_width = 0
                    for feature in hooks.features:
                        max_width = max(max_width, feature.shape[1])

                    padded_features = []
                    for feature in hooks.features:
                        pad_channels = max_width - feature.shape[1]

                        print(feature.shape, pad_channels)

                        if pad_channels > 0:
                            feature = F.pad(feature, (0, pad_channels), "constant", 0)

                        padded_features.append(feature)
                    
                    na = torch.stack(padded_features, dim=-1).numpy()
                    neuronal_abstractions.append(na)

                    hooks.clean_features()
            
                    tbar.update(1)

            neuronal_abstraction_numpy = numpy.concatenate(neuronal_abstractions, axis=0)
            concept_numpy = numpy.concatenate(concepts, axis=0)
            print(neuronal_abstraction_numpy.shape)
            print(concept_numpy.shape)
            numpy.save(f"{dataset_folder}\\{model_name}_{dt}_neuronal_abstractions.npy", neuronal_abstraction_numpy)
            numpy.save(f"{dataset_folder}\\{model_name}_{dt}_concepts.npy", concept_numpy)

    



if __name__ == "__main__":
    generate_neuronal_abstraction_dataset("F:\\causal_pvr_v2\\chain")
    generate_neuronal_abstraction_dataset("F:\\causal_pvr_v2\\collider")
    generate_neuronal_abstraction_dataset("F:\\causal_pvr_v2\\fork")