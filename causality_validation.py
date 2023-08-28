from model.model_training import load_model
import torch
import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import model.Resnet_layers as Resnet_layers
import model.Mobilenet_layers as Mobilenet_layers
import model.Densenet_layers as Densenet_layers

import torch.nn.functional as F
from causality_calculation import single_layer_indicator, multiple_layer_indicator
from train_auxiliary import load_concept_data,build_auxiliary_layer

def do_calculus(explained_model_name, val_dataloader, concept_model_a, concep_index_a, concep_index_b, concep_a_layer_index):


    if os.path.exists("prediction_db.csv"):
        prediction_db = pd.read_csv("prediction_db.csv")
    else:
        prediction_db = pd.DataFrame([],columns=['model','concept','value','predict'])

    concept_vector_a = concept_model_a["1.weight"][concep_index_a,:]
    concept_vector_b = concept_model_a["1.weight"][concep_index_b,:]

    predict = torch.dot(concept_vector_a,concept_vector_b)
    
    if explained_model_name == "resnet":
        explained_model = Resnet_layers.get_model(concep_a_layer_index)
    elif explained_model_name == "densenet":
        explained_model = Densenet_layers.get_model(concep_a_layer_index)
    elif explained_model_name == "mobilenet":
        explained_model = Mobilenet_layers.get_model(concep_a_layer_index)
    else:
        raise Exception(f"Can not process load model {explained_model_name}")
    explained_model.to(torch.device("cuda"))
    explained_model.eval()

    # predict = explained_model(concept_vector_a)


    model = build_auxiliary_layer(concept_model_a["1.weight"].shape[1], concept_model_a["1.weight"].shape[0], "concept_models\\material_classifier_mobilenet_best.pt")
    model.to(torch.device("cuda"))
    model.eval()

    num = 0
    t_sample = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with tqdm.tqdm(total=len(val_dataloader)) as tbar:
        for input_same,annotation in val_dataloader:
            input_same = input_same.to(torch.device("cuda"))
            concept_vector_a = concept_vector_a.to(torch.device("cuda"))

            new_predict = model(input_same + 0.5*concept_vector_a)[0,concep_index_b]
            old_predict = model(input_same)[0,concep_index_b]

            # old_predict = explained_model(input_same)
            # class_index = torch.argmax(old_predict,1)
            # new_predict = explained_model(input_same + 0.5*concept_vector_a)[0,class_index]
            # [0,concep_index_b]

            # print(new_predict, old_predict[0,class_index],predict)
            prediction_db.loc[len(prediction_db.index)] = {'model':"mobilenet",'concept':'material','value':new_predict.data.item()-old_predict.data.item(), 'predict':predict}
            # if torch.sign(predict[class_index]) == torch.sign(new_predict-old_predict[0,class_index]):
            if torch.sign(predict) == torch.sign(new_predict-old_predict):
                t_sample+=1
                tp+=1
            else:
                fp+=1

            num+=1
            tbar.update(1)

    print(f"acc {t_sample/num} {tp}, {fp}")

    prediction_db.to_csv("prediction_db.csv")
        
# def necessity(concept_a, concept_b, dataset_train, dataset_val):


# def sufficiency(concept_a, concept_b, dataset_train, dataset_val):

if __name__ == "__main__":
    explained_model_name = "mobilenet"

    concept_a = torch.load("concept_models\\color_features.12_mobilenet_best.pt")
    concept_c = torch.load("concept_models\\object_classifier_mobilenet_best.pt")
    concept_b = torch.load("concept_models\\part_classifier_mobilenet_best.pt")
    concept_d = torch.load("concept_models\\material_classifier_mobilenet_best.pt")

    _, val_dataloader, input_shape = load_concept_data(model_name=explained_model_name, layer_name="classifier", concept="material", batch_size=1, device=torch.device("cuda"))

    do_calculus(explained_model_name, val_dataloader, concept_d, 4, 5, 4)

    sns.set_theme(style="whitegrid")
    df = pd.read_csv("prediction_db.csv")

    sns.stripplot(
        data=df, x="value", y="concept",
        dodge=True, alpha=.25, zorder=1
    )
    sns.pointplot(
        data=df, x="value", y="concept", 
        join=False, dodge=.8 - .8 / 3, palette="dark",
        markers="d", scale=.75, errorbar=None
    )



    plt.show()
