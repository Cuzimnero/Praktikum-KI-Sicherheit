from pathlib import Path
import shutil
import os

import kagglehub

def download_and_setup_data_set_0():
   download_set("jangedoo/utkface-new")

def download_and_setup_data_set_1():
    download_set("aibloy/fairface")

def download_set(set_name:str):
    path = kagglehub.dataset_download(set_name)
    path_set = Path(path)
    data_path = Path(__file__).parent.parent / "data" / "raw"

    if os.path.exists(path):
        try:
            shutil.move(path_set, data_path)
        except:
            raise RuntimeError("Error while copying files")






if __name__ == "__main__":
    download_and_setup_data_set_0()