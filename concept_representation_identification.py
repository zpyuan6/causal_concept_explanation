from data.pvr_concepts_dataset_generation import PVRConceptDataset
from model.model_training import load_model
from save_feature_maps import ModelHook
from concept_explanation.CAV import CAV
from concept_explanation.Tree_Based_CAV import TREECAV
from concept_explanation.ConceptSHAP import SimplifiedConceptSHAP
from concept_explanation.Concept_Representation import ConceptBasedCausalVariable

import torch
import torch.nn as nn
import torch.utils.data as data_utils

import os
import tqdm
import numpy as np
import copy
import pickle

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

def concept_represententation_identify():
    batch_size = 10
    CONCEPT_LIST = ['TOP_LEFT','TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'IS_ZERO', 'IS_ONE', 'IS_TWO', 'IS_THREE', 'IS_FOUR','IS_FIVE','IS_SIX', 'IS_SEVEN', 'IS_EIGHT','IS_NINE']
    # CONCEPT_LIST = ['BOTTOM_RIGHT', 'IS_ZERO', 'IS_ONE', 'IS_TWO', 'IS_THREE', 'IS_FOUR','IS_FIVE','IS_SIX', 'IS_SEVEN', 'IS_EIGHT','IS_NINE']
    # CONCEPT_LIST = ['IS_EIGHT','IS_NINE']
    NUM_SAMPLE_LIST = [10,20,40,80,160]

    DATASET_NAMES = ['mnist','cifar']
    # DATASET_NAMES = ['mnist']
    # DATASET_NAMES = ['cifar']
    dataset_path = "F:\\pvr_dataset"

    # MODEL_NAMES= ['vgg','resnet','mobilenet','densenet']
    # MODEL_NAMES= ['resnet']
    MODEL_NAMES= ['mobilenet','densenet']

    is_load_saved_data = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    hooks = ModelHook()

    for model_name in MODEL_NAMES:
        for dataset_name in DATASET_NAMES:
            model_path = f"model\pvr_models\{model_name}_{dataset_name}_best.pt"
            model = load_model(model_name, model_path, 10)
           
            model.to(device)
            model.eval()
            for concept_name in CONCEPT_LIST:
                for num_samples in NUM_SAMPLE_LIST:

                    if model_name=="vgg":
                        layers_names = ["features.10","features.20","features.30","features.40","avgpool"]
                        for name,module in model.named_modules():
                            if name in layers_names:
                                hooks.register_hook(model, name,module)
                    elif model_name=="resnet":
                        layers_names = ["maxpool","layer1","layer2","layer3","layer4","avgpool"]
                        # layers_names = ["layer1","layer2","layer3","layer4","avgpool"]
                        # layers_names = ["maxpool"]
                        for name in layers_names:
                            hooks.register_hook(model, name)
                    elif model_name == "mobilenet":
                        layers_names = ["features.3","features.6","features.9","features.12","avgpool"]
                        for name,module in model.named_modules():
                            if name in layers_names:
                                hooks.register_hook(model, name,module)
                    elif model_name=="densenet":
                        layers_names = ["features.denseblock1", "features.denseblock2","features.denseblock3","features.denseblock4"]
                        for name,module in model.named_modules():
                            if name in layers_names:
                                hooks.register_hook(model, name,module)
                    else:
                        raise Exception(f"Can not found modem {model_name}")

                    train_dataloader, val_dataloader= load_dataset(dataset_path, dataset_name, concept_name, num_samples, batch_size)

                    feature_maps = {}
                    labels = None
                    
                    if not is_load_saved_data:
                        with tqdm.tqdm(total=len(train_dataloader)) as tbar:
                            hooks.clean_features()
                            for input, label in train_dataloader:
                                input = input.to(device)
                                output = model(input)
                                labels = label if labels == None else torch.cat((labels, label))
                                for i, feature_map in enumerate(hooks.features):
                                    if layers_names[i] in feature_maps:
                                        feature_maps[layers_names[i]] = torch.cat((feature_maps[layers_names[i]],feature_map), dim=0)
                                    else:
                                        feature_maps[layers_names[i]] = feature_map

                                tbar.update(1)
                                hooks.clean_features()

                    hooks.remove_hooks()

                    for layers_name in layers_names:
                        concepts_path = os.path.join(dataset_path,"concepts_represent")
                        if not os.path.exists(os.path.join(concepts_path,"CAVs")):
                            os.makedirs(os.path.join(os.path.join(concepts_path,"CAVs")))
                        if not os.path.exists(os.path.join(concepts_path,"Tree-based-CAVs")):
                            os.makedirs(os.path.join(os.path.join(concepts_path,"Tree-based-CAVs")))
                        if not os.path.exists(os.path.join(concepts_path,"ConceptSHAP")):
                            os.makedirs(os.path.join(concepts_path,"ConceptSHAP"))
                        if not os.path.exists(os.path.join(concepts_path,"CausalVariables")):
                            os.makedirs(os.path.join(concepts_path,"CausalVariables"))


                        if not is_load_saved_data:
                            features_path = os.path.join(dataset_path, f"{dataset_name}_{model_name}_features")
                            if not os.path.exists(features_path):
                                os.makedirs(features_path) 

                            with open(os.path.join(features_path, f"{concept_name}_{num_samples}_{layers_name}.txt"), "wb") as fp:  
                                pickle.dump((feature_maps[layers_name], labels), fp)
                            # torch.save(feature_maps[layers_name], os.path.join(features_path, f"{concept_name}_{num_samples}_{layers_name}.pt"))
                            print(f"save {dataset_name} {model_name} {concept_name} {num_samples} {layers_name} {feature_maps[layers_name].shape}")
                        else:
                            features_path = os.path.join(dataset_path, f"{dataset_name}_{model_name}_features")
                            with open(os.path.join(features_path, f"{concept_name}_{num_samples}_{layers_name}.txt"), "rb") as fp:  
                                feature_maps[layers_name], labels = pickle.load(fp)
                        # featuremap_dataset = FeatureMapsDataset(feature_maps[layers_name], labels)
                        
                        print(f"Start to identify concept for {dataset_name} {model_name} {layers_name} {concept_name} {num_samples}")
                        
                        # cav = CAV(concept_name, layers_name, {"model_type":'linear',"alpha":0.01}, save_path=os.path.join(concepts_path,"CAVs",f"{dataset_name}_{model_name}_{layers_name}_{concept_name}_{num_samples}.txt"))
                        # cav.train(feature_maps[layers_name],labels)

                        # treecav = TREECAV(concept_name, layers_name, {"model_type":'decisiontree'}, save_path=os.path.join(concepts_path,"Tree-based-CAVs",f"{dataset_name}_{model_name}_{layers_name}_{concept_name}_{num_samples}.txt"))
                        # treecav.train(feature_maps[layers_name],labels)
                        
                        conceptshap = SimplifiedConceptSHAP(n_concepts= len(set(labels.numpy().tolist())), train_embeddings=feature_maps[layers_name], original_model=model, bottleneck=layers_name, example_input_for_original_model=train_dataloader.dataset[0][0], save_path=os.path.join(concepts_path,"ConceptSHAP",f"{dataset_name}_{model_name}_{layers_name}_{concept_name}_{num_samples}.txt"), concept = concept_name)

                        conceptshap.train(feature_maps[layers_name],labels)

                        # concept_represent = ConceptBasedCausalVariable(
                        #     concepts=concept_name, 
                        #     bottleneck=layers_name, 
                        #     hparams={"dimensionality_reduction":"LinearLayer"}, 
                        #     concept_class_num=int(torch.max(labels).item())+1, 
                        #     x = feature_maps[layers_name],
                        #     save_path=os.path.join(concepts_path,"CausalVariables", f"{dataset_name}_{model_name}_{layers_name}_{concept_name}_{num_samples}.txt")
                        #     )
                        # concept_represent.train(feature_maps[layers_name],labels)

                        # concept_represent = ConceptBasedCausalVariable(
                        #     concepts=concept_name, 
                        #     bottleneck=layers_name, 
                        #     hparams={"dimensionality_reduction":"LinearLayer"}, 
                        #     concept_class_num=int(torch.max(labels).item())+1, 
                        #     x = feature_maps[layers_name],
                        #     save_path=os.path.join(concepts_path,"CausalVariables", f"{dataset_name}_{model_name}_{layers_name}_{concept_name}_{num_samples}.txt")
                        #     )
                        # concept_represent.train(feature_maps[layers_name],labels)

def present_and_save_accurate_path(path):

    concept_dict = {}

    for root, folders, files in os.walk(path):
        for file in files:
            if file.split(".")[-1] == 'txt':
                print(f"----------------\nDataset: {file.split('_')[0]};\nModel: {file.split('_')[1]};\nLayer: {file.split('_')[2]};\nConcept: {'_'.join(file.split('_')[3:-1])};\nLength: {file.split('_')[-1]}")

                with open(os.path.join(root, file), 'rb') as pkl_file:
                    results = pickle.load(pkl_file)
                    print(results['concepts'], results['bottleneck'], results['accuracies']['overall'])

            elif file.split(".")[-1] == 'pt':
                print('pt:',file)
                torch_value = torch.load(os.path.join(root, files))
                print(torch_value)



if __name__ == "__main__":
    # concept_represententation_identify()

    path = "F:\\pvr_dataset\\concepts_represent\\CAVs"
    present_and_save_accurate_path(path)
                        
                        




                    



    
    
            



