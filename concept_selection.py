import os
import torch
import torchmetrics
import tqdm
from train_auxiliary import load_concept_data, build_auxiliary_layer


def val_model(model:torch.nn.Module, device, val_dataloader, concept_num, loss_function):
    model.eval()

    test_recall = torchmetrics.Recall(average='none', num_classes=7).to(device)
    test_precision = torchmetrics.Precision(average='none', num_classes=7).to(device)

    total_loss = torch.zeros(concept_num)
    sample_num = torch.zeros(concept_num)
    acc_num = torch.zeros(concept_num)

    TP, FP, TN, FN = torch.zeros(concept_num), torch.zeros(concept_num), torch.zeros(concept_num), torch.zeros(concept_num)
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_dataloader)) as pbar:
            for data, target in val_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                for i in range(output.shape[-1]):
                    loss = loss_function(output[:,i],target[:,i])
                    total_loss[i] += loss.data.cpu()

                sample_num = sample_num + output.shape[0]

                acc_num = acc_num + torch.sum(output.ge(0.5)==target,0).cpu()
                test_recall(output, target)
                test_precision(output, target)

                pbar.update(1)


        total_recall = test_recall.compute()
        total_precision = test_precision.compute()

        print(f"torch metrics acc: {acc_num/sample_num}")
        print("recall of every test dataset class: ", total_recall)
        print("precision of every test dataset class: ", total_precision)

        print("total loss", total_loss)

        # print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avgloss, correct, total_num, 100 * acc))
    
    return total_acc, total_recall, total_precision, total_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    models = ['resnet','mobilenet']

    models_layers = {
        'resnet': ["maxpool","layer1","layer2","layer3","layer4","fc"], 
        'mobilenet': ["features.3","features.6","features.9","features.12","classifier"]
        }

    concept_list = ["color", "material" , "part", "object" ]

    concept_models_path = "concept_models"

    concept_models_list = []

    for root,folders, files in os.walk(concept_models_path):
        for file in files:
            concept_models_list.append(file)
    
    print(f"models: ", concept_models_list)

    loss_function = torch.nn.BCELoss()

    log_file = open('concept_selection.txt', 'a')

    for model_name in models:
        for concept in concept_list:
            # best_layer_name = []
            # best_accuracy = []
            # best_loss = []

            for layer_name in models_layers[model_name]:

                _, val_dataloader, input_shape = load_concept_data(model_name, layer_name, concept, batch_size, device)

                # check model_parameter_path
                model_parameter_file_name = f"{concept}_{layer_name}_{model_name}_best.pt"
                if not model_parameter_file_name in concept_models_list:
                    raise Exception(f"Can not found model parameters {model_parameter_file_name}")

                model = build_auxiliary_layer(input_shape[0], concept_num = len(val_dataloader.dataset.concept_indexs), model_parameter_path=os.path.join(concept_models_path,model_parameter_file_name))

                model.to(device)

                total_acc, total_recall, total_precision, total_loss = val_model(model, device, val_dataloader, len(val_dataloader.dataset.concept_indexs), loss_function)

                print(f"{concept}_{layer_name}_{model_name} total_acc {total_acc}, total_loss {total_loss}, total_recall {total_recall}, total_precision {total_precision}", file=log_file)



                

