import json
import logging
from datetime import datetime
import os
from pathlib import Path

from numpy.ma.extras import average
from ultralytics import YOLO

from src.split_type import split_type,class_type,dataset_type

import utils

import torch.nn as nn
from src.evaluation import evaluator

import torch
from torchvision import datasets, transforms


from torch.utils.data import DataLoader

from src.metrics import mean_absolute_error, cumulative_score
import numpy as np



class model_trainer:
    def __init__(self):
        config=utils.load_config()

        self.main_path = Path(__file__).parent.parent
        self.model_path = self.main_path / config["paths"]["model_path"]
        self.logging_path = self.main_path / config["paths"]["logging_path"]

        self.num_epochs = config["train"]["num_epochs"]
        self.classes_count = config["train"]["classes_count"]
        self.batch_size = config["train"]["batch_size"]
        self.learning_rate = config["train"]["learning_rate"]
        self.num_data_loader_worker = int(config["train"]["num_data_loader_worker"])

        self.log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.logging_path.mkdir(exist_ok=True, parents=True)
        log_file = self.logging_path / self.log_filename
        logging.basicConfig(
            filename=str(log_file),
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger started")

        self.dataset_path = None
        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.loss_function = nn.CrossEntropyLoss()

    def train_default_yolo(self, split: split_type, class_type: class_type,dataset_type:dataset_type, epochs: int, k_fold_value: int,
                        classes_count:int):
        if split is split_type.KFOLD:
            if class_type is class_type.Group:
                if dataset_type is dataset_type.SCALED:
                    self.dataset_path = self.main_path / "data" / "processed" /"scaled"/"k_fold"/"groups"/f"group_size_{classes_count}"

                else:
                    self.dataset_path = self.main_path / "data" / "processed" / "k_fold"/"groups"/f"group_size_{classes_count}"

            elif class_type is class_type.default:
                if dataset_type is dataset_type.SCALED:
                    self.dataset_path=self.main_path / "data" / "processed"/"scaled"/"k_fold"/"default"
                else:
                    self.dataset_path = self.main_path / "data" / "processed" /"k_fold"/ "default"
        else:
            raise TypeError("Invalid split type")

        self.logger.info(f"Using Path {self.dataset_path}")


        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dict_path = self.model_path / "runs" /self.current_dict_name
        run_dict_path.mkdir(parents=True, exist_ok=True)

        for i in range(1, k_fold_value + 1):

            self.default_yolo_model = utils.load_yolo(self.model_path)
            self.yolo_model = self.default_yolo_model.model
            self.yolo_model.train()

            fold_train_path = self.dataset_path / f"fold_{i}" / "train"
            fold_dataset = datasets.ImageFolder(fold_train_path, transform= transforms.ToTensor())
            loader = DataLoader(fold_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_loader_worker)

            in_features = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(in_features, classes_count)


            self.yolo_model.to("cuda")

            for param in self.yolo_model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(self.yolo_model.parameters(), lr=self.learning_rate)

            for epoch in range(0,epochs):
                epoch_loss=0
                for inputs, labels in loader:
                    inputs,labels=inputs.to("cuda"),labels.to("cuda")
                    optimizer.zero_grad()

                    #outputs=self.yolo_model.forward(inputs)
                    outputs=self.yolo_model(inputs)

                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    loss=self.loss_function(outputs, labels)
                    epoch_loss+=loss.item()
                    loss.backward()
                    optimizer.step()

                print(f"Fold {i} Eppoch {epoch} Average Loss: {epoch_loss/len(loader)}")
                self.logger.info(f"Fold {i} Eppoch {epoch} Average Loss: {epoch_loss/len(loader)}")



            torch.save(self.yolo_model.state_dict(),run_dict_path/f"yolo26n-cls_fold{i}.pt")

    def evaluate(self,model:model_trainer,class_count:int,class_type:class_type,k_fold_value:int,name:str):
        eval = evaluator(model.current_dict_name, model.model_path, model.batch_size, model.dataset_path,
                           class_count, model.logger, model.main_path, model.num_data_loader_worker)

        print(f"Average default training accuracy {eval.val_default_yolo(k_fold_value)}")


        if class_type is class_type.Group:
            with open(model.dataset_path / "group-list.json", "r", encoding="utf-8") as file:
                groups = json.load(file)
            group_list = [group["name"] for group in groups.values()]
        else:
            group_list = []
        eval.show_confusion_matrix(group_list, name)
        eval.safe_confusion_matrix()




if __name__ == "__main__":
    model = model_trainer()
    model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.SCALED, epochs=model.num_epochs,
                             k_fold_value=1,classes_count=16)
    model.evaluate(model,16,class_type.Group,1,"Scaled Dataset")


    model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.DEFAULT, epochs=model.num_epochs,
                             k_fold_value=1,classes_count=16)
    model.evaluate(model,16,class_type.Group,1,"Unscaled Dataset")
    #
    # model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.DEFAULT, epochs=model.num_epochs,
    #                          k_fold_value=1, group_count=4,classes_count=4)
    # model.evaluate(model,4,class_type.Group,1,"Unscaled Dataset GroupSize 4")
    #
    #
    # model.train_default_yolo(split_type.KFOLD, class_type.Group,dataset_type.DEFAULT, epochs=model.num_epochs, k_fold_value=1,
    #                          group_count=10,classes_count=10)
    #
    # model.evaluate(model,10,class_type.Group,1,"Unscaled Dataset GroupSize 10")
    #
    #
    #
    # model.train_default_yolo(split_type.KFOLD, class_type.Group,dataset_type.DEFAULT, epochs=model.num_epochs, k_fold_value=1,
    #                          group_count=12,classes_count=12)
    # model.evaluate(model,12,class_type.Group,1,"Unscaled Dataset GroupSize 12")
    #



