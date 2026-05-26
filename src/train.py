import json
import logging
from datetime import datetime
import os
from pathlib import Path

from ultralytics import YOLO

from src.split_type import split_type,class_type


import torch.nn as nn
from src.evaluation import evaluator

import torch
from torchvision import datasets, transforms


from torch.utils.data import DataLoader



class model_trainer:
    def __init__(self):
        self.log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.loss_function = nn.CrossEntropyLoss()
        self.num_epochs = 10
        self.main_path = Path(__file__).parent.parent

        self.logging_path = self.main_path/"logs"
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

        self.model_path = self.main_path / "models"
        self.dataset_path = None
        self.classes_count=99
        self.batch_size = 256
        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.learning_rate = 0.001

    def train_default_yolo(self, split:split_type,class_type:class_type, epochs:int, k_fold_value:int,group_count:int ):
        if split is split_type.KFOLD:
            if class_type is class_type.Group:
                self.dataset_path = self.main_path / "data" / "processed" / "k_fold"/"groups"/f"group_size_{group_count}"
            elif class_type is class_type.default:
                self.dataset_path = self.main_path / "data" / "processed" /"k_fold"/ "default"
        else:
            raise TypeError("Invalid split type")


        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dict_path = self.model_path / "runs" /self.current_dict_name
        run_dict_path.mkdir(parents=True, exist_ok=True)

        for i in range(1, k_fold_value + 1):

            self.default_yolo_model = YOLO(self.model_path / "yolo26n-cls.pt")
            self.yolo_model = self.default_yolo_model.model
            self.yolo_model.train()

            fold_train_path = self.dataset_path / f"fold_{i}" / "train"
            fold_dataset = datasets.ImageFolder(fold_train_path, transform= transforms.ToTensor())
            loader = DataLoader(fold_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

            in_features = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(in_features, self.classes_count)


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




if __name__ == "__main__":
    model = model_trainer()
    model.train_default_yolo(split_type.KFOLD,class_type.Group,epochs=model.num_epochs,k_fold_value=1,group_count=4)
    evaluator = evaluator(model.current_dict_name, model.model_path, model.batch_size, model.dataset_path,model.classes_count,model.logger,model.main_path)
    print(f"Average default training accuracy {evaluator.val_default_yolo(1 ,4)}")
    with open(model.dataset_path/"group-list.json", "r", encoding="utf-8") as file:
        groups = json.load(file)
    group_list = [group["name"] for group in groups.values()]
    evaluator.show_confusion_matrix(group_list)
    evaluator.safe_confusion_matrix()

    model.train_default_yolo(split_type.KFOLD, class_type.Group, epochs=model.num_epochs, k_fold_value=1, group_count=10)
    evaluator.update_dataset_path(model.dataset_path)
    print(f"Average default training accuracy {evaluator.val_default_yolo(1,10)}")
    with open(model.dataset_path/"group-list.json", "r", encoding="utf-8") as file:
        groups = json.load(file)
    group_list = [group["name"] for group in groups.values()]
    evaluator.show_confusion_matrix(group_list)
    evaluator.safe_confusion_matrix()

    model.train_default_yolo(split_type.KFOLD, class_type.Group, epochs=model.num_epochs, k_fold_value=1, group_count=12)
    evaluator.update_dataset_path(model.dataset_path)
    print(f"Average default training accuracy {evaluator.val_default_yolo(1,12)}")
    with open(model.dataset_path/"group-list.json", "r", encoding="utf-8") as file:
        groups = json.load(file)
    group_list = [group["name"] for group in groups.values()]
    evaluator.show_confusion_matrix(group_list)
    evaluator.safe_confusion_matrix()




