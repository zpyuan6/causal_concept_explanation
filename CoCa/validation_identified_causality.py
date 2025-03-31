import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from CoCa.module.CausalConceptVAE import CausalConceptVAE
from utils.plot import plot_causal_adjacency_matrix
from model.model_training import load_model
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from data.PVRDataset import CausalPVRDataset
from CoCa.dataset.NeuronalAbstraction import NeuronalAbstractionDataset


def visualise_causal_adjacency_matrix_in_VAE_probe(path):
    # model_name = ['resnet', 'mobilenet']
    model_name = ['resnet']

    for name in model_name:
        probe_path = os.path.join(path, f"probe_{name}_train","CausalConceptVAE_best.pth")
        probe = CausalConceptVAE(model_file_path = probe_path)

        adjacency_matrix = probe.get_causal_adjacency_matrix()

        concept_locations, concept_representations_for_each_concept, concept_representations_for_each_concept_value = probe.get_concept_representations()

        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()

        plot_causal_adjacency_matrix(adjacency_matrix)
        print(adjacency_matrix)
        print(concept_locations)


def get_causal_effect_matrix(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = ['resnet']

    batch_size = 128

    for name in model_name:
        training_data = NeuronalAbstractionDataset(path, name, "train")
        train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

        probe_path = os.path.join(path, f"probe_{name}_train","CausalConceptVAE_best.pth")
        probe = CausalConceptVAE(model_file_path = probe_path)
        probe.eval()
        probe.to(device)

        causal_structure = probe.get_causal_adjacency_matrix()

        concept_locations, concept_representations_for_each_concept, concept_representations_for_each_concept_value = probe.get_concept_representations()
        
        causal_effect_matrix = torch.zeros_like(causal_structure)
        number_treated_samples = torch.zeros(causal_structure.shape[0]).to(device)
        
        for batch_idx, items in enumerate(train_dataloader):
            x = items[0].to(device)
            concept_label = items[2].to(device)

            for i, concept_dim in enumerate(probe.concept_dims):
                for concept_value in range(concept_dim):
                    selected_sample_indexes = concept_label[:, i] != concept_value
                    selected_treated_samples = x[selected_sample_indexes]
                    treated_operation = torch.zeros_like(selected_treated_samples)
                    treated_operation[:, :,concept_locations[i]] +=  (2 * concept_representations_for_each_concept_value[i][:,concept_value]).unsqueeze(0).repeat(selected_treated_samples.shape[0],1)
                    treatment_samples = selected_treated_samples - treated_operation

                    controlled_result = concept_label[selected_sample_indexes]
                    mu, log_var, concept_classes = probe.encode(treatment_samples)
                    concept_classes = torch.cat([torch.argmax(F.softmax(concept_class, dim = -1), dim=-1).unsqueeze(-1) for concept_class in concept_classes], dim=-1)

                    change_items = (concept_classes != controlled_result)

                    num_changed_items = change_items.sum(dim=0)

                    target_effect = causal_structure[i] > 0.004

                    causal_effect_matrix[i] += (num_changed_items * target_effect)
                    number_treated_samples[i] += selected_sample_indexes.sum()
                    
            print(causal_effect_matrix)

                # target_concept = causal_effect_matrix[i]

                # concept_layer_location = concept_locations[i]
                
                # concept_label_for_concept_i = concept_label[:, i]

                # operator_concept_value = torch.ones((x.shape[0], concept_dim))

                # operator_concept_value[concept_label_for_concept_i] = 0
                
                # concept_operat_value = []

                # do_calculate_samples = {f"concept_{i}": concept_label_for_concept_i for i in range(len(concept_label_for_concept))}

if __name__ == "__main__":
    # visualise_causal_adjacency_matrix_in_VAE_probe("F:\\causal_pvr_v2\\collider")

    get_causal_effect_matrix("F:\\causal_pvr_v2\\collider")