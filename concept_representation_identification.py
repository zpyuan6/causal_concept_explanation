from data.pvr_concepts_dataset_generation import PVRConceptDataset, FeatureMapsDataset
from model.model_training import load_model
from save_feature_maps import ModelHook
from concept_explanation.CAV import CAV
from concept_explanation.Tree_Based_CAV import TREECAV
from concept_explanation.ConceptSHAP import ConceptSHAP

import torch
import torch.nn as nn
import torch.utils.data as data_utils

import os
import tqdm
import numpy as np
import copy

def load_dataset(dataset_path, dataset_name, concept_name, num_samples, batch_size):
    
    train_dataset_path = os.path.join(dataset_path, f"{dataset_name}_concept", f"{dataset_name}_{concept_name}_{num_samples}.txt")
    val_dataset_path = os.path.join(dataset_path, f"{dataset_name}_concept", f"val_{dataset_name}_{concept_name}_{num_samples}.txt")
    
    train_dataset = PVRConceptDataset(train_dataset_path)
    val_dataset = PVRConceptDataset(val_dataset_path)

    train_dataloader = data_utils.DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
    val_dataloader = data_utils.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

    return train_dataloader, val_dataloader

# def train_cav_svm(input_features, concept_labels):
#     featuremap_dataset = FeatureMapsDataset(input_features, concept_labels)


# def train_concept_represent_svm(input_features, concept_labels):
#   



if __name__ == "__main__":
    batch_size = 10
    CONCEPT_LIST = ['TOP_LEFT','TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'IS_ZERO', 'IS_ONE', 'IS_TWO', 'IS_THREE', 'IS_FOUR','IS_FIVE','IS_SIX', 'IS_SEVEN', 'IS_EIGHT','IS_NINE']
    NUM_SAMPLE_LIST = [10,20,40,80,160]

    DATASET_NAMES = ['mnist','cifar']
    dataset_path = "F:\\pvr_dataset"

    MODEL_NAMES= ['vgg','resnet','mobilenet','densenet']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hooks = ModelHook()

    for model_name in MODEL_NAMES:
        for dataset_name in DATASET_NAMES:
            model_path = f"model\pvr_models\{model_name}_{dataset_name}_best.pt"
            model = load_model(model_name, model_path, 10)
           

            if model_name=="vgg":
                layers_names = ["features.10","features.20","features.30","features.40","classifier"]
                for name,module in model.named_modules():
                    if name in layers_names:
                        hooks.register_hook(model, name,module)
            elif model_name=="resnet":
                layers_names = ["maxpool","layer1","layer2","layer3","layer4","fc"]
                for name in layers_names:
                    hooks.register_hook(model, name)
            elif model_name == "mobilenet":
                layers_names = ["features.3","features.6","features.9","features.12","classifier"]
                for name,module in model.named_modules():
                    if name in layers_names:
                        hooks.register_hook(model, name,module)
            elif model_name=="densenet":
                layers_names = ["features.denseblock1", "features.denseblock2","features.denseblock3","features.denseblock4","classifier"]
                for name,module in model.named_modules():
                    if name in layers_names:
                        print("-----",name)
                        hooks.register_hook(model, name,module)
            else:
                raise Exception(f"Can not found modem {model_name}")

            model.to(device)
            for concept_name in CONCEPT_LIST:
                for num_samples in NUM_SAMPLE_LIST:
                    train_dataloader, val_dataloader= load_dataset(dataset_path, dataset_name, concept_name, num_samples, batch_size)

                    feature_maps = {}
                    labels = None
                    
                    with tqdm.tqdm(total=len(train_dataloader)) as tbar:
                        for input, label in train_dataloader:
                            input = input.to(device)
                            output = model(input)
                            labels = label if labels == None else torch.cat((labels, label))

                            for i, feature_map in enumerate(hooks.features):
                                if layers_names[i] in feature_maps:
                                    feature_maps[layers_names[i]] = torch.cat((feature_maps[layers_names[i]],feature_map), dim=0)
                                else:
                                    feature_maps[layers_names[i]] = feature_map

                            hooks.clean_features()
                            tbar.update(1)

                    

                    for layers_name in feature_maps.keys():
                        if not os.path.exists(os.path.join(dataset_path,"CAVs")):
                            os.makedirs(os.path.join(os.path.join(dataset_path,"CAVs")))
                        if not os.path.exists(os.path.join(dataset_path,"Tree-based-CAVs")):
                            os.makedirs(os.path.join(os.path.join(dataset_path,"Tree-based-CAVs")))
                        
                        cav = CAV(concept_name, layers_name, {"model_type":'linear',"alpha":0.01}, save_path=os.path.join(dataset_path,"CAVs",f"{dataset_name}_{model_name}_{layers_name}_{concept_name}.txt"))
                        cav.train(feature_maps[layers_name],labels)

                        treecav = TREECAV(concept_name, layers_name, {"model_type":'decisiontree'}, save_path=os.path.join(dataset_path,"Tree-based-CAVs",f"{dataset_name}_{model_name}_{layers_name}_{concept_name}.txt"))
                        treecav.train(feature_maps[layers_name],labels)



                        features_path = os.path.join(dataset_path, f"{dataset_name}_{model_name}_features")
                        if not os.path.exists(features_path):
                            os.makedirs(features_path) 
                        torch.save(feature_maps[layers_name], os.path.join(features_path, f"{concept_name}_{num_samples}_{layers_name}.pt"))
                        print(f"save {dataset_name} {model_name} {concept_name} {num_samples} {layers_name} {feature_maps[layers_name].shape}")

                        featuremap_dataset = FeatureMapsDataset(feature_maps[layers_name], labels)




                    



    
    
            



