from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, Notears, GraNDAG
import pandas as pd
import numpy as np

def pc(path):
    concepts = np.load(path + "\\train_concept_casaul_pvr.npy")
    weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10, 
                                      weight_range=(0.5, 2.0), seed=1)
    true_causal_matrix = np.array(
        [[0,0,0,0,1,0,0],
         [0,0,0,0,1,0,0],
         [0,0,0,0,0,0,1],
         [0,0,0,0,0,0,1],
         [0,0,0,0,0,0,1],
         [0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0]]
    )

    pc = PC()
    pc.learn(concepts)

    notears = Notears()
    notears.learn(concepts)

    graNDAG = GraNDAG(input_dim=7)
    graNDAG.learn(concepts)

    GraphDAG(pc.causal_matrix, true_causal_matrix, 'result')
    GraphDAG(notears.causal_matrix, true_causal_matrix, 'result')
    GraphDAG(graNDAG.causal_matrix, true_causal_matrix, 'result')

    # calculate metrics
    mt = MetricsDAG(pc.causal_matrix, true_causal_matrix)
    print(mt.metrics)



if __name__ == "__main__":
    path = "F:\\causal_pvr_v2\\collider"
    pc(path)
