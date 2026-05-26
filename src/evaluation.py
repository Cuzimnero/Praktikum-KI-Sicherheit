from datetime import datetime
import logging
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class evaluator:
    def __init__(self, current_dict_name:str, model_path:Path, batch_size:int, dataset_path:Path, classes_count:int, logger: logging,main_path:Path):
        self.cm = None
        self.current_dict_name = current_dict_name
        self.batch_size = batch_size
        self.model_path = model_path
        self.default_yolo_model = None
        self.yolo_model = None
        self.dataset_path = dataset_path
        self.loss_function = nn.CrossEntropyLoss()
        self.classes_count = classes_count
        self.logger = logger
        self.main_path = main_path

    def val_default_yolo(self,k_fold_value:int,group_count:int):
        print("starting evaluation")
        test_loss = 0
        accuracy_scores=[]

        for i in range(1, k_fold_value + 1):
            predicted_classes = []
            actual_classes = []

            if self.dataset_path is None:
                raise ValueError("Dataset path not initialized")

            fold_eval_path = self.dataset_path / f"fold_{i}" / "val"
            eval_dataset = datasets.ImageFolder(fold_eval_path, transform= transforms.ToTensor())
            loader=DataLoader(eval_dataset,batch_size=self.batch_size,num_workers=6)

            self.default_yolo_model = YOLO(self.model_path / "yolo26n-cls.pt")
            self.yolo_model = self.default_yolo_model.model

            in_features = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(in_features, self.classes_count)
            self.yolo_model.to("cuda")

            try:
                weights = torch.load(self.model_path / "runs" / self.current_dict_name / f"yolo26n-cls_fold{i}.pt")
                self.yolo_model.load_state_dict(weights)
            except FileNotFoundError:
                raise FileNotFoundError("Model not found")

            self.yolo_model.eval()

            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                    #outputs=self.yolo_model.forward(inputs)
                    outputs = self.yolo_model(inputs)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    loss=self.loss_function(outputs, labels)
                    test_loss+=loss.item()
                    predicted_classes.extend(torch.argmax(outputs, 1).cpu().numpy())
                    actual_classes.extend(labels.cpu().numpy())

            fold_accuracy=accuracy_score(actual_classes,predicted_classes)
            self.cm=confusion_matrix(actual_classes,predicted_classes)
            print(f"Fold {i} : accuracy {fold_accuracy}")
            self.logger.info(f"Fold {i} : accuracy {fold_accuracy}")
            accuracy_scores.append(fold_accuracy)

        average=sum(accuracy_scores)/k_fold_value
        self.logger.info(f"Average default training accuracy {average} Group count: {group_count}")
        return average

    def update_dataset_path(self,dataset_path:Path):
            self.dataset_path = dataset_path

    def show_confusion_matrix(self,display_labels:list):
        matrix=ConfusionMatrixDisplay(confusion_matrix=self.cm,display_labels=display_labels).plot()
        self.figure=matrix.figure_
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def safe_confusion_matrix(self):
        file_name="confusion_matrix_"+datetime.now().strftime("%Y-%m-%d_%H-%M") + ".png"
        result_dir = self.main_path / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.main_path /"results"/file_name )


