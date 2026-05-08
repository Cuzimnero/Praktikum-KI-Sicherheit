import os
from pathlib import Path

from ultralytics import YOLO

from src import split_type


def train_default_yolo(type:split_type,epochs:int,imgsz:(200,200),k_fold_value:int):

    main_path=Path(__file__).parent.parent
    model_path=main_path / "models"
    if type is split_type.split_type.KFOLD:
        dataset_path = main_path / "data" / "processed" / "k_fold"
    else:
        raise TypeError("Invalid split type")
    model = YOLO(model_path / "yolo26n-cls.pt")
    for i in range(1,k_fold_value+1):
        fold_train_path=dataset_path /f"fold_{i}"/"train"
        #TODO



if __name__ == "__main__":
    train_default_yolo(split_type.split_type.KFOLD,epochs=40,imgsz=(200,200),k_fold_value=10)



