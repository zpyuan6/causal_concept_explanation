import pandas as pd

def statistic_scen_class(dataset_form):
    print(dataset_form)
    
    groups = dataset_form.groupby('scene_category_name')

    print(len(groups))
    for key, value in groups:
        print(key)

if __name__ == "__main__":
    annotation_path = "F:\\Broden\\opensurfaces_photos_label.csv"

    opensurfer_dataset = pd.read_csv(annotation_path)

    statistic_scen_class(opensurfer_dataset)
