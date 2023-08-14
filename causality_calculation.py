from model.model_training import load_model
import torch

import model.Resnet_layers as Resnet_layers
import model.Mobilenet_layers as Mobilenet_layers
import model.Densenet_layers as Densenet_layers

import torch.nn.functional as F

def single_layer_indicator(concept_model_a:torch.Tensor, concep_index_a:int, concept_model_b:torch.Tensor, concep_index_b:int):
    

    concept_vector_a = F.normalize(concept_model_a[concep_index_a,:],p=2,dim=0)

    concept_vector_b = F.normalize(concept_model_b[concep_index_b,:],p=2,dim=0)

    # print(concept_vector_a)

    return torch.dot(concept_vector_a, concept_vector_b)


def multiple_layer_indicator(concept_model_a:torch.nn.Module, concep_index_a:int, concep_a_layer_index:int, concept_model_b:torch.nn.Module, concep_index_b:int, explained_model_name:str):
    # layer a < layer b

    concept_vector_a = concept_model_a[concep_index_a,:]
    
    if explained_model_name == "resnet":
        model = Resnet_layers.get_model(concep_a_layer_index)
    elif explained_model_name == "densenet":
        model = Densenet_layers.get_model(concep_a_layer_index)
    elif explained_model_name == "mobilenet":
        model = Mobilenet_layers.get_model(concep_a_layer_index)
    else:
        raise Exception(f"Can not process load model {explained_model_name}")

    model.eval()

    # concept_vector_a = torch.reshape(concept_vector_a, [1,96, 7,7]).cpu()
    concept_vector_a = concept_vector_a.cpu()

    transefered_concept_verctor_a = F.normalize(torch.flatten(model(concept_vector_a)),p=2,dim=0) 

    # print(transefered_concept_verctor_a)

    concept_vector_b = F.normalize(concept_model_b[concep_index_b,:].cpu(),p=2,dim=0) 
    # concept_vector_b = torch.nn.functional.one_hot(torch.tensor(1))
    print(transefered_concept_verctor_a)

    return torch.dot(transefered_concept_verctor_a, concept_vector_b)


if __name__ == "__main__":
    explained_model_name = "mobilenet"
    explained_model_parameter_path = f"model\logs\{explained_model_name}_best.pt"

    explained_model = load_model(explained_model_name, explained_model_parameter_path)

    models_layers = {
        "densenet": ["features.denseblock1", "features.denseblock2","features.denseblock3","features.denseblock4","classifier"],
        'resnet': ["maxpool","layer1","layer2","layer3","layer4","fc"], 
        'mobilenet': ["features.3","features.6","features.9","features.12","classifier"]
        }

    concept_num = [11, 12, 12, 13]
    # concept_list = ["color_fc_resnet_best.pt","material_fc_resnet_best.pt","object_fc_resnet_best.pt","part_fc_resnet_best.pt"]

    concept_list = ["color_features.12_mobilenet_best.pt","material_classifier_mobilenet_best.pt","object_classifier_mobilenet_best.pt","part_classifier_mobilenet_best.pt"]
    # concept_list = []

    concept_a = torch.load("concept_models\\color_features.12_mobilenet_best.pt")
    concept_b = torch.load("concept_models\\material_classifier_mobilenet_best.pt")
    concept_c = torch.load("concept_models\\object_classifier_mobilenet_best.pt")
    concept_d = torch.load("concept_models\\part_classifier_mobilenet_best.pt")
    print(concept_a)

    for param in concept_a:

        print(param, concept_a[param].shape)

    # print(single_layer_indicator(concept_c["1.weight"],3,concept_c["1.weight"],0))
    # print(single_layer_indicator(concept_c["1.weight"],3,concept_c["1.weight"],1))
    # print(single_layer_indicator(concept_c["1.weight"],3,concept_c["1.weight"],5))
    # print(single_layer_indicator(concept_c["1.weight"],3,concept_c["1.weight"],7))
    # print(single_layer_indicator(concept_c["1.weight"],3,concept_c["1.weight"],8))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],1))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],2))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],3))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],4))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],5))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],6))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],7))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],8))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],9))
    # print(single_layer_indicator(concept_a["1.weight"],0,concept_a["1.weight"],10))
    # print("-------------------------")
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),1))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),2))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),3))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),4))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),5))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),6))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),7))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),8))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),9))
    # print(single_layer_indicator(F.normalize(concept_a["1.weight"], p=2, dim=1),0,F.normalize(concept_a["1.weight"], p=2, dim=1),10))
    
    # print("-------------------------")
    # print(multiple_layer_indicator(concept_a["1.weight"],0,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],0,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],1,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],1,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],2,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],2,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],3,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],3,3,concept_b["1.weight"],5,explained_model_name))

    # print(multiple_layer_indicator(concept_a["1.weight"],4,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],4,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],5,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],5,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],6,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],6,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],7,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],7,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],8,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],8,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],9,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],9,3,concept_b["1.weight"],5,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],10,3,concept_b["1.weight"],4,explained_model_name))
    # print(multiple_layer_indicator(concept_a["1.weight"],10,3,concept_b["1.weight"],5,explained_model_name))



    print(multiple_layer_indicator(concept_c["1.weight"],8,4,concept_b["1.weight"],4,explained_model_name))
    # print("-------------------------")
    # print(single_layer_indicator(concept_b["1.weight"],4,concept_c["1.weight"],0))
    # print(single_layer_indicator(concept_b["1.weight"],4,concept_d["1.weight"],0))
    # print(single_layer_indicator(concept_b["1.weight"],5,concept_c["1.weight"],0))
    # print(single_layer_indicator(concept_b["1.weight"],5,concept_d["1.weight"],0))

    # print(single_layer_indicator(concept_b["1.weight"],4,concept_c["1.weight"],6))
    # print(single_layer_indicator(concept_b["1.weight"],4,concept_d["1.weight"],1))
    # print(single_layer_indicator(concept_b["1.weight"],5,concept_c["1.weight"],6))
    # print(single_layer_indicator(concept_b["1.weight"],5,concept_d["1.weight"],1))


    # construct concepts
    # concepts = {}
    # for concept in concept_list:
    #     concept_name, layer_name, model = concept.split("_")

    #     if layer_name in concepts:
    #         concepts[layer_name].append(concept)
    #     else:
    #         concepts[layer_name] = [concept]

    # for layer_name in concepts.keys():
    #     for 

    

    
