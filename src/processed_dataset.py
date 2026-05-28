import json
import shutil as sh
import pathlib
import splitfolders as fs
from src.split_type import split_type,class_type

data_dict_path = pathlib.Path(__file__).parent.parent/"data"
group_map={}
buckets = list()


def split_dataset_k_fold(dataset:pathlib.Path,result_path:pathlib.Path,k:int):
    fs.kfold(dataset,result_path,seed=42,k=k,move= "symlink")

def process_dataset(dataset:pathlib.Path,result_path:pathlib.Path,k:int):
    if not dataset.exists():
        return
    files=dataset.glob("*.jpg")
    for file in files:
        informations=file.name.split("_")
        sub_dict = (result_path / informations[0])
        if not int(informations[0]) in buckets:
            buckets.append(int(informations[0]))
            sub_dict.mkdir(exist_ok=True,parents=True)
        sh.copy(file, sub_dict)
    k_fold_sort_path=result_path/".."/".."/"k_fold"/"default"
    k_fold_sort_path.mkdir(exist_ok=True,parents=True)
    split_dataset_k_fold(result_path,k_fold_sort_path,k)
    
    for folder in k_fold_sort_path.rglob("*"):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

def process_dataset_groups(dataset:pathlib.Path,result_path:pathlib.Path,k:int,group_count:int):
    if not dataset.exists():
        return
    k_fold_sort_path = result_path/".."/".."/"k_fold"/"groups"/f"group_size_{group_count}"
    k_fold_sort_path.mkdir(exist_ok=True,parents=True)
    files = dataset.glob("*.jpg")
    create_groups(group_count,k_fold_sort_path)
    for file in files:
        information=file.name.split("_")
        for group in group_map.values():
            sub_dict = (result_path /f"group_size_{group_count}"/ group["name"])
            if int(information[0])>int(group["end"]) or int(information[0])<int(group["start"]):
                continue
            sub_dict.mkdir(exist_ok=True, parents=True)
            sh.copy(file, sub_dict)
            break
    split_dataset_k_fold(result_path/f"group_size_{group_count}", k_fold_sort_path, k)

    for folder in k_fold_sort_path.rglob("*"):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()



def create_groups(group_count:int,dest_dir:pathlib.Path):
    if not buckets:
        raise RuntimeError("No buckets found")

    sorted_buckets = sorted(buckets)
    group_size = len(sorted_buckets) / group_count
    for i in range(0, group_count):
        group_start = sorted_buckets[int(i * group_size)]
        group_end = sorted_buckets[int((i+1) * group_size)-1]
        group_map[i] = {
            "start": group_start, "end": group_end,"name":f"{group_start}-{group_end}"
        }
    with open(dest_dir/"group-list.json", "w", encoding="utf-8") as file:
        json.dump(group_map, file, ensure_ascii=False, indent=4)

def equalize_data(data_set_path:pathlib.Path,split_type:split_type,class_type:class_type,k_fold:int,class_count:int):
        data_path=data_set_path /"train"
        k_fold_v=0
        if split_type is split_type.KFOLD:
            if class_type is class_type.Group:
                dict=data_dict_path/"processed"/"scaled"/"k_fold"/"groups"/f"group_size_{class_count}"
            else:
                dict=data_dict_path/"processed"/"scaled"/"k_fold"/"default"
            k_fold_v=k_fold
        else:
            dict = data_dict_path / "processed" / "scaled"

        file_count=0
        sub_count=0
        for fold in range(1,k_fold_v+1):
            if split_type is split_type.KFOLD:
                working_dict = dict / f"fold_{fold}"/"train"
                working_dict.mkdir(exist_ok=True,parents=True)
                data_path = data_set_path / f"fold_{fold}" / "train"

            for sub_dir in data_path.iterdir():
                sub_count = sub_count + 1
                if sub_dir.is_dir():
                    file_count = file_count + sum(1 for file in sub_dir.iterdir() if file.is_file())

            average_file_per_class = file_count / sub_count

            print(file_count)
            print("Average number of files: ", average_file_per_class)
            print(sub_count)
            file_count = 0
            sub_count = 0

            for sub_dir in data_path.iterdir():
                class_path = working_dict / sub_dir.name
                class_path.mkdir(exist_ok=True, parents=True)
                if sub_dir.is_dir():
                    for file in sub_dir.iterdir():
                        file_count = 0
                        file_count = file_count + 1
                        if file_count > average_file_per_class:
                            break
                        sh.copy(file, class_path)
                    if file_count < average_file_per_class:
                        it = 0
                        while (file_count < average_file_per_class):
                            it = it + 1
                            for file in class_path.iterdir():
                                sh.copy(file, class_path / f"{file.stem}({it}).jpg")
                                file_count = file_count + 1
            sh.copy(data_set_path/"group-list.json",dict)
            if split_type is split_type.KFOLD:
                data_path = data_set_path / f"fold_{fold}" / "val"
                working_dict = dict / f"fold_{fold}"/"val"
            else:
                data_path = data_set_path / f"fold_{fold}" / "val"
                working_dict = dict / "val"
            for sub_dir in data_path.iterdir():
                class_path = working_dict / sub_dir.name
                class_path.mkdir(exist_ok=True, parents=True)
                for file in sub_dir.iterdir():
                    sh.copy(file, class_path)

        try:
            sh.copy(data_set_path/"group-list.json",dict)
        except:
            raise FileNotFoundError("Group list file not found")
















if __name__ == "__main__":
    process_dataset(data_dict_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_dict_path/"processed"/"default"/"default",5)
    process_dataset_groups(data_dict_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_dict_path/"processed"/"default"/"groups",5,16)
    equalize_data(data_dict_path/"processed"/"k_fold"/"groups"/"group_size_16",split_type.KFOLD,class_type.Group,5,16)
    #equalize_data(data_dict_path/"processed"/"k_fold"/"default","default",split_type.KFOLD,5)
    #process_dataset(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"default",5)
    #process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,4)
    #process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,10)
    #process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,12)



