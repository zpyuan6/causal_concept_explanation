import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from CoCa.utils.training_utils import AugementedLagrangianTrainer
from CoCa.module.CausalConceptVAE import CausalConceptVAE
from CoCa.module.CausalConceptVAEv2 import CausalConceptVAEv2
from CoCa.dataset.NeuronalAbstraction import NeuronalAbstractionDataset


def optimise_global_VAE_probe(path):

    # model_name = ['resnet', 'mobilenet']
    model_name = ['resnet']

    for name in model_name:

        # training_data = NeuronalAbstractionDataset(path, name, "train", concept_remove_y=False)
        # test_data = NeuronalAbstractionDataset(path, name, "test", concept_remove_y=False)
        # probe = CausalConceptVAEv2(
        #     input_shape=training_data.get_input_shape(),
        #     concept_dims=training_data.get_concept_dims(),
        #     concept_remove_y=True
        # )

        training_data = NeuronalAbstractionDataset(path, name, "train", concept_remove_y=True)
        test_data = NeuronalAbstractionDataset(path, name, "test", concept_remove_y=True)
        probe = CausalConceptVAE(
            input_shape=training_data.get_input_shape(),
            concept_dims=training_data.get_concept_dims(),
            concept_remove_y=True
        )

        dataset_name = path.split('\\')[-1]
        trainer = AugementedLagrangianTrainer(
            model = probe,
            training_dataset=training_data,
            test_dataset=test_data,
            batch_size=128,
            learning_rate=1e-3,
            num_epochs=200,
            num_workers=6,
            model_save_path = f'{path}\\probe_{name}_train\\',
            dataset_name = dataset_name
        )
        trainer.augemented_lagrangian_training()


if __name__ == "__main__":
    optimise_global_VAE_probe("F:\\causal_pvr_v2\\collider")