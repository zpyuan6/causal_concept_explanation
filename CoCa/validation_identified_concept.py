import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from CoCa.module.CausalConceptVAE import CausalConceptVAE


def visualise_causal_graphs_in_VAE_probe(path):
    probe = CausalConceptVAE(model_file_path=path)

    num_concepts = len(probe.concept_dims)

    concept_representations_for_each_value, concept_locations = probe.get_concept_representations()

    concept_representations_for_each_concept = []
    
    for i in range(num_concepts):
        concept_representations_for_each_concept.append(
            concept_representations_for_each_value[i].cpu().detach().numpy().sum(dim=0)
        )

    

if __name__ == "__main__":
    visualise_causal_graphs_in_VAE_probe("F:\\causal_pvr_v2\\collider")