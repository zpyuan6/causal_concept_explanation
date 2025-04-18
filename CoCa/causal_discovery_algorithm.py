from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, Notears, GraNDAG
import pandas as pd
import numpy as np

def pc(path):
    all_concepts = np.load(path + "\\train_concept_casaul_pvr.npy")

    acc = []

    for i in range(10):

        random_indices = np.random.choice(all_concepts.shape[0], size=100)
        concepts = all_concepts[random_indices]

        # weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10, 
        #                                 weight_range=(0.5, 2.0), seed=1)
        # collider
        true_causal_matrix = np.array(
            [[0,0,0,0,1,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]]
        )

        # chain
        # true_causal_matrix = np.array(
        #     [[0,0,0,0,1,0,0],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,1],
        #     [0,0,0,0,0,0,1],
        #     [0,0,0,0,0,0,1],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0]]
        # )

        # fork
        # true_causal_matrix = np.array(
        #     [[0,0,0,0,1,0,0],
        #     [0,0,0,0,1,0,0],
        #     [0,0,0,0,0,0,1],
        #     [0,0,0,0,0,0,1],
        #     [0,0,0,0,0,1,1],
        #     [0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0]]
        # )

        # pc = PC()
        # pc.learn(concepts)
        # GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

        # pc = Notears()
        # pc.learn(concepts)
        # GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

        pc = GraNDAG(input_dim=7)
        pc.learn(concepts)
        GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')

        # calculate metrics
        # print(pc.causal_matrix)
        # print(true_causal_matrix)

        # mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
        # print(mt.metrics)

        acc.append(np.sum(pc.causal_matrix==true_causal_matrix).item()/49)

    print(f"acc,", acc)
    print(f"average,", sum(acc)/len(acc))
    print(f"std,", np.std(acc))


if __name__ == "__main__":
    path = "F:\\causal_pvr_v2\\collider"
    # path = "F:\\causal_pvr_v2\\fork"
    # path = "F:\\causal_pvr_v2\\chain"
    pc(path)
