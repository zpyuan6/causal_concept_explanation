from model.model_training import load_model

def single_layer_indicator(concept_model_a, concept_model_b):


def multiple_layer_indicator(concept_model_a, concept_model_b, explained_model):

if __name__ == "__main__":
    explained_model_name = "resnet"
    explained_model_parameter_path = f"model\logs\{explained_model_name}_best.pt"

    explained_model = load_model(explained_model_name, explained_model_parameter_path)

    models_layers = {
        'resnet': ["maxpool","layer1","layer2","layer3","layer4","fc"], 
        'mobilenet': ["features.3","features.6","features.9","features.12","classifier"]
        }

    concept_list = []

    for concept_a in concept_list:

    
