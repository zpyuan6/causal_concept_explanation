import os
from train_auxiliary import load_concept_data, build_auxiliary_layer


if __name__ == "__main__":
    models = ['resnet']

    models_layers = {
        'vgg': [],
        'resnet': ["maxpool","layer1","layer2","layer3","layer4","fc"], 
        'mobilenet': []
        }

    concept_list = ["color", "material" , "part", "object" ]

    concept_models_path = "concept_models"

    concept_models_list = []

    for root,folders, files in os.walk(concept_models_path):
        for file in files:
            concept_models_list.append(file)
    
    print(f"models: ", concept_models_list)

    for model_name in models:
        for concept in concept_list:
            for layer_name in models_layers[model_name]:

                _, val_dataloader, input_shape = load_concept_data(model_name, layer_name, concept)

                # check model_parameter_path
                model_parameter_file_name = f"{concept}_{layer_name}_{model_name}_best.pt"
                if not model_parameter_file_name in concept_models_list:
                    raise Exception(f"Can not found model parameters {model_parameter_file_name}")

                model = build_auxiliary_layer(input_shape[0], concept_num = len(val_dataloader.dataset.concept_indexs), model_parameter_path=os.path.join(concept_models_path,model_parameter_file_name)) 

