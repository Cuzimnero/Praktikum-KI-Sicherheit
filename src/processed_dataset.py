import shutil as sh
import pathlib
import splitfolders as fs

data_path = pathlib.Path(__file__).parent.parent/"data"

def split_dataset_k_fold(dataset:pathlib.Path,result_path:pathlib.Path,k:int):
    fs.kfold(dataset,result_path,seed=42,k=k,move= "symlink")

def process_dataset(dataset:pathlib.Path,result_path:pathlib.Path,k:int):
    if not dataset.exists():
        return
    files=dataset.glob("*.jpg")
    buckets=set()
    for file in files:
        informations=file.name.split("_")
        sub_dict = (result_path / informations[0])
        if not informations[0] in buckets:
            buckets.add(informations[0])
            sub_dict.mkdir(exist_ok=True,parents=True)
        sh.copy(file, sub_dict)
    k_fold_sort_path=result_path/".."/"k_fold"
    k_fold_sort_path.mkdir(exist_ok=True)
    split_dataset_k_fold(result_path,k_fold_sort_path,k)

    for folder in k_fold_sort_path.rglob("*"):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


if __name__ == "__main__":
    process_dataset(data_path/"raw"/"1"/"utkface_aligned_cropped"/"crop_part1",data_path/"processed"/"default",5)

