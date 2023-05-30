import pandas as pd
import os
import shutil
import random

if __name__ == "__main__":
    base_path = "F:\Broden"

    annotation_path = f"{base_path}\\opensurfaces_photos_label.csv"
    source_folder = f"{base_path}\\images\\opensurfaces"
    target_folder = f"{base_path}\\opensurfaces\\train"

    dataset = pd.read_csv(annotation_path)

    # for index, row in dataset.iterrows():
    #     print(row['photo_id'],row['scene_category_name'])
    #     source_file = os.path.join(source_folder, f"{row['photo_id']}.jpg")

    #     target_file = os.path.join(target_folder,row['scene_category_name'] ,f"{row['photo_id']}.jpg")

    #     if not os.path.exists(os.path.join(target_folder,row['scene_category_name'])):
    #         os.makedirs(os.path.join(target_folder,row['scene_category_name']))
        
    #     shutil.copyfile(source_file, target_file)

    target_val_path = f"{base_path}\\opensurfaces\\val"

    for r, folders, _ in os.walk(target_folder):
        for folder in folders:
            print(folder)
            for root, _, files in os.walk(os.path.join(r,folder)):
                if not os.path.exists(os.path.join(target_val_path,folder)):
                    os.makedirs(os.path.join(target_val_path,folder))
                val_samples = random.sample(files, int(len(files)*0.1) if len(files)*0.1>=1 else 1)
                for file in val_samples:
                    shutil.move(os.path.join(root,file), os.path.join(target_val_path,folder,file))
                

