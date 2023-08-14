# causal_concept_explanation

## How to use

- Training and validation dataset preparing
- Explained model training
- Explained model evaluation
- Concept dataset preparing
- Concept model training (Concept discovered)
- Causal relationship calculation
- Causal relationship verification

## How to start

### PVR tasks

1. Dataset Generation for PVR
In this stage, two PVR　dataset is generated
```
python data\pvr_dataset_generation.py
```

2. Explained Model Training
In this stage, we will training 4 model on two dataset including PVR_MNIST and PVR_CIFAR10. 
```
python model_training_for_pvr_task.py
```

3. PVR Concept Dataset Generation
In this stage, PVR Concept Datasets are generated.
```
python data\pvr_concepts_dataset_generation.py
```