from datetime import datetime
import os
from pathlib import Path

from ultralytics import YOLO

from src import split_type
from sklearn.metrics import accuracy_score

import torch.nn as nn

import torch
from torchvision import datasets, transforms


from torch.utils.data import DataLoader



class model_trainer:
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()
        self.num_epochs = 5
        self.main_path = Path(__file__).parent.parent
        self.model_path = self.main_path / "models"
        self.dataset_path = None
        self.classes_count=99
        self.batch_size = 256
        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.learning_rate = 0.001

    def train_default_yolo(self, split:split_type, epochs:int, k_fold_value:int, ):
        if split is split_type.split_type.KFOLD:
            self.dataset_path = self.main_path / "data" / "processed" / "k_fold"
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
            self.yolo_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
            fold_dataset = datasets.ImageFolder(fold_train_path, transform=self.yolo_transforms)
            loader = DataLoader(fold_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

            # in_features = self.yolo_model.model[-1].linear.in_features
            # self.yolo_model.model[-1].linear = nn.Linear(in_features, self.classes_count)


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


            torch.save(self.yolo_model.state_dict(),run_dict_path/f"yolo26n-cls_fold{i}.pt")

    def val_default_yolo(self,k_fold_value:int):
        print("starting evaluation")
        test_loss = 0
        accuracy_scores=[]

        for i in range(1, k_fold_value + 1):
            predicted_classes = []
            actual_classes = []

            fold_eval_path = self.dataset_path / f"fold_{i}" / "val"
            eval_dataset = datasets.ImageFolder(fold_eval_path, transform=self.yolo_transforms)
            loader=DataLoader(eval_dataset,batch_size=self.batch_size,num_workers=6)

            self.default_yolo_model = YOLO(self.model_path / "yolo26n-cls.pt")
            self.yolo_model = self.default_yolo_model.model

            #in_features = self.yolo_model.model[-1].linear.in_features
            #self.yolo_model.model[-1].linear = nn.Linear(in_features, self.classes_count)
            self.yolo_model.to("cuda")

            try:
                weights = torch.load(self.model_path / "runs" / self.current_dict_name / f"yolo26n-cls_fold{i}.pt")
                self.yolo_model.load_state_dict(weights,strict=False)
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

            print(f"{actual_classes}, \n {predicted_classes}")
            fold_accuracy=accuracy_score(actual_classes,predicted_classes)
            print(f"Fold {i} : accuracy {fold_accuracy}")
            accuracy_scores.append(fold_accuracy)



        return sum(accuracy_scores)/k_fold_value





if __name__ == "__main__":
    model = model_trainer()
    model.train_default_yolo(split_type.split_type.KFOLD,epochs=model.num_epochs,imgsz=(200,200),k_fold_value=1)
    print(f"Average default training accuracy {model.val_default_yolo(1)}")



