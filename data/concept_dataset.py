import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

CONCEPT_TYPE = ['color','material','part','object']

COLOR_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MATERIAL_INDEX = [0,1,2,128,23,33,47,50,186,59,192,68]
PART_INDEX = [0, 1, 2, 4, 6, 8, 9,14, 17, 20, 21, 22]
OBJECT_INDEX = [0, 1, 2, 3, 6, 11, 12, 13, 14, 15, 17, 18, 20]

CONCEPT_LEN  = [11, 35, 234, 584]
class ConceptDataset(data.Dataset):
    def __init__(self, model_name, layer_name, device, concept_type:str='color',train_or_val:str="train", input_path:str = "F:\\Broden\\concept_model\\feature_maps", annotation_path:str="F:\\Broden\\concept_model\\concept_annotation_processed") -> None:
        
        self.concept_type = concept_type

        if concept_type == CONCEPT_TYPE[0]:
            self.concept_indexs = COLOR_INDEX
        elif concept_type == CONCEPT_TYPE[1]:
            self.concept_indexs = MATERIAL_INDEX
        elif concept_type == CONCEPT_TYPE[2]:
            self.concept_indexs = PART_INDEX
        elif concept_type == CONCEPT_TYPE[3]:
            self.concept_indexs = OBJECT_INDEX
        else:
            raise Exception(f"Can not found concept {concept_type}")

        annotation_path = os.path.join(annotation_path,train_or_val)
        self.annotation_list = []
        self.input_list = []
        self.device = device

        for root, folders, files in os.walk(annotation_path):
            for file in files:
                if root.split("\\")[-1] == self.concept_type:
                    annotation_from_file = json.load(open(os.path.join(root, file), 'r'))
                    annotation_list = set(self.concept_indexs).intersection(set(annotation_from_file))
                    if len(annotation_list)>0:
                        annotation = torch.zeros(len(self.concept_indexs))
                        for concept_index in annotation_list:
                            annotation[self.concept_indexs.index(concept_index)] = 1
                        annotation = annotation.to(self.device)
                        self.annotation_list.append(annotation)
                        self.input_list.append(os.path.join(input_path,f"{model_name}_{layer_name}",train_or_val,root.split("\\")[-2],file.split(".")[0]+".pt"))

    def __getitem__(self, index):
        input_tensor = torch.load(self.input_list[index], map_location=self.device)
        input_tensor = torch.flatten(input_tensor)

        annotation = self.annotation_list[index]

        return input_tensor, annotation

    def __len__(self):
        return len(self.input_list)


def statistic_concept_number():
    dataset = ConceptDataset(concept_type="color")
    print("train color samples: ", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    dataset = ConceptDataset(concept_type="color",train_or_val='val')
    print("val color samples: ", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)

    dataset = ConceptDataset(concept_type="material")
    print("train material", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    dataset = ConceptDataset(concept_type="material",train_or_val='val')
    print("val material", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)

    dataset = ConceptDataset(concept_type="part")
    print("train part", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    dataset = ConceptDataset(concept_type="part",train_or_val='val')
    print("val part", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)

    dataset = ConceptDataset(concept_type="object")
    print("train object", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    dataset = ConceptDataset(concept_type="object",train_or_val='val')
    print("val object", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)


if __name__ == "__main__":
    statistic_concept_number()