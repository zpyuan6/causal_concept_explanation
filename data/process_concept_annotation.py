import os
from skimage import io
import numpy as np
import json
import shutil

def get_item_list():

    dataset_folder = "F:\Broden\opensurfaces"
    train_folder = os.path.join(dataset_folder,"train")
    val_folder = os.path.join(dataset_folder,"val")
    ade20k_list, opensurface_list = [], []

    annotation_folder = "F:\\Broden\\opensurfaces\\concept_annotation"

    for root, folders, files in os.walk(train_folder):
        for file in files:
            print(os.path.join(root,file))
            folder = root.split("\\")[-1]
            if "ADE" in file:
                concept_path = "F:\\Broden\\images\\ade20k"
                color_concept,material_concept,part_concept,object_concept = get_concepts(concept_path, file)
                ade20k_list.append({
                        "train_or_val": "train",
                        "type":folder,
                        "image": file,
                        "color_concept": color_concept,
                        "material_concept": material_concept,
                        "part_concept": part_concept,
                        "object_concept": object_concept
                    })

                if not os.path.exists(os.path.join(annotation_folder, "train", folder)):
                    os.makedirs(os.path.join(annotation_folder, "train", folder))
                
                annotation_json = json.dumps({
                    "color_concept": color_concept,
                    "material_concept": material_concept,
                    "part_concept": part_concept,
                    "object_concept": object_concept
                }) 

                annotation_file = open(os.path.join(annotation_folder, "train", folder, file.split(".")[0]+".json"),'w')
                annotation_file.write(annotation_json)
                annotation_file.close()
            else:
                continue
                concept_path = "F:\Broden\images\opensurfaces"
                color_concept,material_concept,part_concept,object_concept = get_concepts(concept_path, file)
                opensurface_list.append({
                        "train_or_val": "train",
                        "type":folder,
                        "image": file,
                        "color_concept": color_concept,
                        "material_concept": material_concept,
                        "part_concept": part_concept,
                        "object_concept": object_concept
                })

                if not os.path.exists(os.path.join(annotation_folder, "train", folder)):
                    os.makedirs(os.path.join(annotation_folder, "train", folder))
                
                annotation_json = json.dumps({
                    "color_concept": color_concept,
                    "material_concept": material_concept,
                    "part_concept": part_concept,
                    "object_concept": object_concept
                }) 

                annotation_file = open(os.path.join(annotation_folder, "train", folder, file.split(".")[0]+".json"),'w')
                annotation_file.write(annotation_json)
                annotation_file.close()
                    


    for root, folders, files in os.walk(val_folder):
        for file in files:
            folder = root.split("\\")[-1]
            if "ADE" in file:
                concept_path = "F:\\Broden\\images\\ade20k"
                color_concept,material_concept,part_concept,object_concept = get_concepts(concept_path, file)
                ade20k_list.append({
                        "train_or_val": "val",
                        "type":folder,
                        "image": file,
                        "color_concept": color_concept,
                        "material_concept": material_concept,
                        "part_concept": part_concept,
                        "object_concept": object_concept
                })

                if not os.path.exists(os.path.join(annotation_folder, "val", folder)):
                    os.makedirs(os.path.join(annotation_folder, "val", folder))
                
                annotation_json = json.dumps({
                    "color_concept": color_concept,
                    "material_concept": material_concept,
                    "part_concept": part_concept,
                    "object_concept": object_concept
                }) 

                annotation_file = open(os.path.join(annotation_folder, "val", folder, file.split(".")[0]+".json"),'w')
                annotation_file.write(annotation_json)
                annotation_file.close()
            else:
                continue
                concept_path = "F:\Broden\images\opensurfaces"
                color_concept,material_concept,part_concept,object_concept = get_concepts(concept_path, file)
                opensurface_list.append({
                        "train_or_val": "val",
                        "type":folder,
                        "image": file,
                        "color_concept": color_concept,
                        "material_concept": material_concept,
                        "part_concept": part_concept,
                        "object_concept": object_concept
                })

                if not os.path.exists(os.path.join(annotation_folder, "val", folder)):
                    os.makedirs(os.path.join(annotation_folder, "val", folder))
                
                annotation_json = json.dumps({
                    "color_concept": color_concept,
                    "material_concept": material_concept,
                    "part_concept": part_concept,
                    "object_concept": object_concept
                }) 

                annotation_file = open(os.path.join(annotation_folder, "val", folder, file.split(".")[0]+".json"),'w')
                annotation_file.write(annotation_json)
                annotation_file.close()

    annotation_file_ade20k = json.dumps(ade20k_list)
    annotation_file_opensurface = json.dumps(opensurface_list)

    all_ade20k_annotation = open("ade20k.json",'w')
    all_ade20k_annotation.write(annotation_file_ade20k)
    all_ade20k_annotation.close()

    all_opensurface_annotation = open("opensurface.json",'w')
    all_opensurface_annotation.write(annotation_file_opensurface)
    all_opensurface_annotation.close()

    return ade20k_list, opensurface_list

def get_concepts(dataset, input_file):
    color_concept,material_concept,part_concept,object_concept = [],[],[],[]
    sample_id = input_file.split(".")[0]
    print(f"-------------------Find concept for {input_file}")
    for root, folders, files in os.walk(dataset):
        for file in files:
            if str(file).find(sample_id+"_") == 0:
                concept_type = file.split(".")[0].split("_")[-1]

                img = io.imread(os.path.join(root, file))
                annotation = np.unique(img).tolist()
                annotation.remove(0)
                annotation = [i-1 for i in annotation]
                print(concept_type, annotation)

                if concept_type =='color':
                    color_concept.extend(annotation)
                elif concept_type =='material':
                    material_concept.extend(annotation)
                elif concept_type =='object':
                    object_concept.extend(annotation)
                elif concept_type =='part' or concept_type =='1' or concept_type=='2':
                    part_concept.extend(annotation)
                else:
                    print(f"Do not have {concept_type}")

    return color_concept,material_concept,part_concept,object_concept


def concept_annotation_reorganisation():
    dataset_annotation_path = "F:\\Broden\\concept_model\\concept_annotation"

    for root, folders, files in os.walk(dataset_annotation_path):
        for file in files:
            annotation = json.load(open(os.path.join(root,file), 'r'))

            path_arr = root.split("\\")
            path_arr[3] = "concept_annotation_processed"
            dataset_annotation_save_path = "\\".join(path_arr)
            color_folder = os.path.join(dataset_annotation_save_path, "color")
            material_folder = os.path.join(dataset_annotation_save_path, "material")
            part_folder = os.path.join(dataset_annotation_save_path, "part")
            object_folder = os.path.join(dataset_annotation_save_path, "object")

            if not os.path.exists(color_folder):
                os.makedirs(color_folder)
                os.makedirs(material_folder)
                os.makedirs(part_folder)
                os.makedirs(object_folder)
            
            if len(annotation["color_concept"])>0:
                anno = json.dumps(annotation["color_concept"])
                f = open(os.path.join(color_folder,file), 'w')
                f.write(anno)
                f.close()

            if len(annotation["material_concept"])>0:
                anno = json.dumps(annotation["material_concept"])
                f = open(os.path.join(material_folder,file), 'w')
                f.write(anno)
                f.close()

            if len(annotation["part_concept"])>0:
                anno = json.dumps(annotation["part_concept"])
                f = open(os.path.join(part_folder,file), 'w')
                f.write(anno)
                f.close()

            if len(annotation["object_concept"])>0:
                anno = json.dumps(annotation["object_concept"])
                f = open(os.path.join(object_folder,file), 'w')
                f.write(anno)
                f.close()


if __name__ == "__main__":
    # ade20k_list, opensurface_list = get_item_list()
    concept_annotation_reorganisation()