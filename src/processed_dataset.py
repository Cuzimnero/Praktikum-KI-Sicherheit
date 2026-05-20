import shutil as sh
import pathlib
import splitfolders as fs

data_path = pathlib.Path(__file__).parent.parent/"data"
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
    create_groups(group_count)
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



def create_groups(group_count:int):
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





if __name__ == "__main__":
    process_dataset(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"default",5)
    process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,4)
    process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,10)
    process_dataset_groups(data_path/"raw"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default"/"groups",5,12)



