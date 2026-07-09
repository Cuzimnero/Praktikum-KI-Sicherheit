from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src import utils
from src.metrics import cumulative_score
from src.split_type import train_type


class evaluator:
    def __init__(self, current_dict_name:str, model_path:Path, batch_size:int, dataset_path:Path, classes_count:int, logger: logging,main_path:Path,num_data_loader_worker:int,train_type:train_type):
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
        self.num_data_loader_worker = num_data_loader_worker
        self.weights_path=None
        self.train_type = train_type

    def val_default_yolo(self,k_fold_value:int):
        print("starting evaluation")

        accuracy_scores=[]
        self.cm=[]
        mae_scores=[]
        cs_scores=[]

        for i in range(1, k_fold_value + 1):
            if self.train_type is train_type.default:
                self.weights_path = self.model_path / "runs" / self.current_dict_name / f"yolo26n-cls_fold{i}.pt"
            else:
                self.weights_path = self.model_path / "runs" / self.current_dict_name / f"yolo26n-cls_distillation_fold{i}.pt"

            predicted_classes = []
            actual_classes = []
            test_loss=0

            if self.dataset_path is None:
                raise ValueError("Dataset path not initialized")

            fold_eval_path = self.dataset_path / f"fold_{i}" / "val"
            eval_dataset = datasets.ImageFolder(fold_eval_path, transform= transforms.ToTensor())
            loader=DataLoader(eval_dataset,batch_size=self.batch_size,num_workers=self.num_data_loader_worker)

            self.default_yolo_model = utils.load_yolo(self.model_path)
            self.yolo_model = self.default_yolo_model.model

            in_features = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(in_features, self.classes_count)
            self.yolo_model.to("cuda")

            try:
                weights = torch.load(self.weights_path)
                self.yolo_model.load_state_dict(weights)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model not found: {e}")

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
            self.cm.append(confusion_matrix(actual_classes,predicted_classes))

            y_predicted = np.array(predicted_classes)
            y_real = np.array(actual_classes)

            fold_mae = mean_absolute_error(y_real, y_predicted)
            fold_cs = cumulative_score(y_real, y_predicted, tolerance=1)

            print(f"Fold {i} : accuracy {fold_accuracy} | MAE: {fold_mae:.2f} | CS (±1): {fold_cs * 100:.1f}%")
            self.logger.info(f"Fold {i} : accuracy {fold_accuracy} | MAE: {fold_mae:.2f} | CS (±1): {fold_cs * 100:.1f}%")

            accuracy_scores.append(fold_accuracy)
            mae_scores.append(fold_mae)
            cs_scores.append(fold_cs)

        average_accuracy=sum(accuracy_scores)/k_fold_value
        average_mae=sum(mae_scores)/k_fold_value
        average_cs=sum(cs_scores)/k_fold_value


        self.logger.info(f"Average default training accuracy {average_accuracy} | Average MAE: {average_mae:.2f} | Average CS: {average_cs * 100:.1f}% Group count: {self.classes_count} ")
        return {"accuracy": average_accuracy, "mae":average_mae, "cs":average_cs}


    def show_confusion_matrix(self,display_labels:list,name:str):
        self.figure =[]
        for confusion_matrix in self.cm:
            matrix=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=display_labels).plot()
            fig=matrix.figure_
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=45, ha='right')
            fig.suptitle(name, y=0.05, fontsize=10, style="italic")

            self.figure.append(fig)
            plt.show()

    def safe_confusion_matrix(self):
        result_dir = self.main_path / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        for index, fig in enumerate(self.figure):
            file_name = f"confusion_matrix_fold_{index + 1}_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + ".png"
            fig.savefig(result_dir / file_name, bbox_inches='tight')


