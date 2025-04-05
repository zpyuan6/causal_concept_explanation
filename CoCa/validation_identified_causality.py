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
from tqdm import tqdm

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
        training_data = NeuronalAbstractionDataset(path, name, "train", concept_remove_y=False)
        train_dataloader = DataLoader(training_data, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)
        concept_dims = training_data.get_concept_dims()

        probe_path = os.path.join(path, f"probe_{name}_train","CausalConceptVAE_best.pth")
        probe = CausalConceptVAE(model_file_path = probe_path)
        probe.eval()
        probe.to(device)

        causal_structure = probe.get_causal_adjacency_matrix()
        concept_locations, concept_representations_for_each_concept, concept_representations_for_each_concept_value = probe.get_concept_representations()

        causal_structure_mask = causal_structure > 0.004
        
        causal_effect_matrix = torch.zeros_like(causal_structure_mask).float()
        number_treated_samples = torch.zeros(causal_structure_mask.shape[0]).to(device)
        
        with tqdm(total=len(train_dataloader), desc="Calculating causal effect matrix", unit="batch") as pbar:
            for batch_idx, items in enumerate(train_dataloader):
                x = items[0].to(device)
                y = items[1].to(device)
                concept_label = items[2].to(device)

                for i, concept_dim in enumerate(concept_dims):
                    for concept_value in range(concept_dim):
                        selected_sample_indexes = concept_label[:, i] != concept_value
                        selected_treated_samples = x[selected_sample_indexes]
                        treated_operation = torch.zeros_like(selected_treated_samples)

                        if i == len(concept_dims) - 1:
                            treated_operation[:, :,-1] += (2 * F.one_hot(torch.tensor(concept_value), num_classes=selected_treated_samples.shape[1]).to(selected_treated_samples.device)).unsqueeze(0).repeat(selected_treated_samples.shape[0],1) 
                        else:
                            treated_operation[:, :,concept_locations[i]] +=  (2 * concept_representations_for_each_concept_value[i][:,concept_value]).unsqueeze(0).repeat(selected_treated_samples.shape[0],1)
                        treatment_samples = selected_treated_samples - treated_operation

                        mu, log_var, concept_classes = probe.encode(treatment_samples)

                        z = [probe.reparameterize(mu[i], log_var[i]) for i in range(len(probe.concept_dims))]
                        concept_classes = [F.softmax(concept_class, dim = -1) for concept_class in concept_classes]
                        concept_classes_index = torch.cat([torch.argmax(concept_class, dim=-1).unsqueeze(-1) for concept_class in concept_classes], dim=-1)
                        controlled_result = concept_label[selected_sample_indexes, :concept_classes_index.shape[1]]
                        change_items = (concept_classes_index != controlled_result)

                        num_changed_items = change_items.sum(dim=0)

                        recons = probe.decode(z, concept_classes, treatment_samples)
                        y_pred = recons[:, :concept_dims[-1], -1]
                        y_pred = torch.argmax(y_pred, dim=-1)
                        y_change = (y_pred != y[selected_sample_indexes]).sum(dim=0)

                        num_changed_items = torch.cat([num_changed_items, y_change.unsqueeze(0)], dim=0)

                        target_effect = causal_structure_mask[i]

                        causal_effect_matrix[i] += (num_changed_items * target_effect)
                        number_treated_samples[i] += selected_sample_indexes.sum()

                pbar.update(1)

        for i in range(causal_effect_matrix.shape[0]):
            causal_effect_matrix[i] /= number_treated_samples[i]

        final_causal_effect_matrix = causal_effect_matrix.cpu().detach().numpy()

        plot_causal_adjacency_matrix(final_causal_effect_matrix)

        print(final_causal_effect_matrix)
        
        return final_causal_effect_matrix



def prune_causal_structure(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = ['resnet']

    for name in model_name:
        training_data = NeuronalAbstractionDataset(path, name, "train", concept_remove_y=False)
        concept_numpy = training_data.get_concept_labels()

        probe_path = os.path.join(path, f"probe_{name}_train","CausalConceptVAE_best.pth")
        probe = CausalConceptVAE(model_file_path = probe_path)
        probe.eval()
        probe.to(device)

        adjacency_matrix = probe.get_causal_adjacency_matrix()
        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()

        plot_causal_adjacency_matrix(adjacency_matrix)

        adjacency_matrix = probe.prune_causal_structure(concept_numpy)

        plot_causal_adjacency_matrix(adjacency_matrix)
    

if __name__ == "__main__":
    # visualise_causal_adjacency_matrix_in_VAE_probe("F:\\causal_pvr_v2\\collider")

    # get_causal_effect_matrix("F:\\causal_pvr_v2\\collider")

    prune_causal_structure("F:\\causal_pvr_v2\\collider")