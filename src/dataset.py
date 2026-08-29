from pathlib import Path
import shutil
import os

import kagglehub

def download_and_setup_data_set_0():
    """Herunterladen des UTK-Face Datasets in den data Ordner"""
    download_set("jangedoo/utkface-new")

def download_and_setup_data_set_1():
    """Herunterladen des Fairface Datasets in den data Ordner"""
    download_set("aibloy/fairface")

def download_set(set_name:str):
    """Lädt ein Datenset in die Testumgebung
        Parameter
        ----------
        set_name : str
            Name des Datenset
         Raises
        ------
        RuntimeError
            Falls installations-Pfad  nicht existiert / Kopierfehler auftritt
        """
    path = kagglehub.dataset_download(set_name)
    path_set = Path(path)
    data_path = Path(__file__).parent.parent / "data" / "raw"
    data_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(path):
        try:
            shutil.move(path_set, data_path)
        except:
            raise RuntimeError("Error while copying files")






if __name__ == "__main__":
    download_and_setup_data_set_0()