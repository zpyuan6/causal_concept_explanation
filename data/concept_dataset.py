import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt

CONCEPT_TYPE = ['color','material','part','object']
CONCEPT_LEN  = [11, 35, 234, 584]
class ConceptDataset(data.Dataset):
    def __init__(self, input_path:str = "F:\\Broden\\opensurfaces", annotation_path:str="F:\\Broden\\opensurfaces\\concept_annotation", train_or_val:str="train", concept_type:str='color', transforms:transforms=None) -> None:
        annotation_path = os.path.join(annotation_path,train_or_val)
        input_path = os.path.join(input_path,train_or_val)

        self.transforms = transforms
        self.annotation_list = []
        self.image_list = []
        self.concept_type = concept_type
        self.concept_indexs = set()

        if not concept_type in CONCEPT_TYPE:
            raise Exception(f"Can not found concept {concept_type}")

        for root, folders, files in os.walk(annotation_path):
            for file in files:
                annotation = json.load(open(os.path.join(root, file), 'r'))
                if len(annotation[concept_type+"_concept"])>0:
                    self.annotation_list.append(annotation[concept_type+"_concept"])
                    self.concept_indexs.update(annotation[concept_type+"_concept"])
                    class_type = root.split("\\")[-1]
                    self.image_list.append(os.path.join(input_path,class_type,file.split(".")[0]+".jpg"))

        self.concept_indexs = [i for i in self.concept_indexs]

        for i, concept_annotation in enumerate(self.annotation_list):
            annotation = torch.zeros(len(self.concept_indexs))
            for concept_index in concept_annotation:
                annotation[self.concept_indexs.index(concept_index)] = 1
            self.annotation_list[i] = annotation
                

    def __getitem__(self, index):

        img = Image.open(self.image_list[index]).convert('RGB')
        print(type(img))
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.annotation_list[index]

    def __len__(self):
        return len(self.image_list)


def statistic_concept_number():
    dataset = ConceptDataset(concept_type="color")
    print("color samples: ", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    dataset.__getitem__(3)
    # dataset = ConceptDataset(concept_type="material")
    # print("material", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    # dataset = ConceptDataset(concept_type="part")
    # print("part", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)
    # dataset = ConceptDataset(concept_type="object")
    # print("object", dataset.__len__(), "types: ", len(dataset.concept_indexs), dataset.concept_indexs)


if __name__ == "__main__":
    statistic_concept_number()