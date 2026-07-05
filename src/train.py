import json
import logging
from datetime import datetime
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mivolo_outer"))
MIVOLO_ROOT = PROJECT_ROOT / "src" / "mivolo_outer"
if str(MIVOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(MIVOLO_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
#from unittest import loader

import torch.nn.functional as F

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


import mivolo_outer.mivolo.model.mivolo_model as mvm


class MiVOLOTrainer(nn.Module):
    def __init__(self, weights_path, classes_count=16, sigma=4.0):
        super().__init__()

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        print("Checkpoint Informationen:")
        print(f"min_age = {checkpoint['min_age']}")
        print(f"max_age = {checkpoint['max_age']}")
        print(f"avg_age = {checkpoint['avg_age']}")
        print(f"no_gender = {checkpoint['no_gender']}")
        print()

        self.min_age = float(checkpoint["min_age"])
        self.max_age = float(checkpoint["max_age"])
        self.avg_age = float(checkpoint["avg_age"])

        self.only_age = bool(checkpoint["no_gender"])
        output_classes = 1 if self.only_age else 3

        self.mivolo = mvm.MiVOLOModel(
            layers=(4, 4, 8, 2),
            embed_dims=(192, 384, 384, 384),
            num_classes=output_classes,
            num_heads=(6, 12, 12, 12),
            img_size=224,
            in_chans=3
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        model_dict = self.mivolo.state_dict()

        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        print(f"Geladene Parameter: {len(pretrained_dict)} / {len(model_dict)}")

        model_dict.update(pretrained_dict)
        self.mivolo.load_state_dict(model_dict, strict=False)

        for p in self.mivolo.parameters():
            p.requires_grad = False

        self.mivolo.eval()

        self.sigma = sigma

        class_centers = torch.tensor([
            3.5,
            15.5,
            21.5,
            27.5,
            34.0,
            40.5,
            46.5,
            52.5,
            58.5,
            65.0,
            71.5,
            9.5,
            77.5,
            83.5,
            89.5,
            101.5
        ], dtype=torch.float32)

        self.register_buffer("class_centers", class_centers)

        self._printed = False

    def forward(self, x):

        x = (x - self.mean) / self.std

        with torch.no_grad():
                out = self.mivolo(x)

        if isinstance(out, tuple):
            out = out[0]

        if self.only_age:
            age = out[:, 0]
        else:

            age = out[:, 2]

        age = age * (self.max_age - self.min_age) + self.avg_age

        if not self._printed:
            print("=" * 60)
            print("Erste 20 Altersvorhersagen von MiVOLO:")
            print(age[:20])
            print("=" * 60)
            self._printed = True

        diff = age.unsqueeze(1) - self.class_centers.unsqueeze(0)

        teacher_logits = -(diff ** 2) / (2 * self.sigma ** 2)

        return teacher_logits

class EmbeddingExtractor:

        def __init__(self, model: nn.Module, layer_name: str):
            self.model = model
            self.layer_name = layer_name
            self.embedding = None
            self.hook = None
            self._register_hook()

        def _get_layer(self):
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    return module

            available_layers = [name for name, _ in self.model.named_modules()]
            raise ValueError(
                f"Layer '{self.layer_name}' wurde nicht gefunden.\n"
                f"Verfügbare Layer:\n{available_layers}"
            )

        def _hook_fn(self, module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]

            self.embedding = output

        def _register_hook(self):
            layer = self._get_layer()
            self.hook = layer.register_forward_hook(self._hook_fn)
            print(f"Hook registriert auf Layer: {self.layer_name} ({layer.__class__.__name__})")

        def get_embedding(self):
            if self.embedding is None:
                raise RuntimeError(
                    f"Noch kein Embedding für Layer '{self.layer_name}' gespeichert. "
                    f"Erst Forward Pass ausführen."
                )

            return self.embedding

        def remove(self):
            if self.hook is not None:
                self.hook.remove()
                self.hook = None

def pool_embedding(x):


        if isinstance(x, (tuple, list)):
            x = x[0]

        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        elif x.dim() == 3:
            x = x.mean(dim=1)

        elif x.dim() == 2:
            pass

        else:
            raise ValueError(f"Unbekannte Feature-Shape: {x.shape}")

        return x




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
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

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
            fold_dataset = datasets.ImageFolder(fold_train_path, transform= self.transform)
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

    def train_distillation_yolo(
            self,
            split: split_type,
            class_type: class_type,
            dataset_type: dataset_type,
            epochs: int,
            k_fold_value: int,
            classes_count: int
    ):
        if split is split_type.KFOLD:
            if class_type is class_type.Group:
                if dataset_type is dataset_type.SCALED:
                    self.dataset_path = (
                            self.main_path / "data" / "processed" / "scaled" /
                            "k_fold" / "groups" / f"group_size_{classes_count}"
                    )
                else:
                    self.dataset_path = (
                            self.main_path / "data" / "processed" /
                            "k_fold" / "groups" / f"group_size_{classes_count}"
                    )

            elif class_type is class_type.default:
                if dataset_type is dataset_type.SCALED:
                    self.dataset_path = (
                            self.main_path / "data" / "processed" /
                            "scaled" / "k_fold" / "default"
                    )
                else:
                    self.dataset_path = (
                            self.main_path / "data" / "processed" /
                            "k_fold" / "default"
                    )
        else:
            raise TypeError("Invalid split type")

        self.logger.info(f"Using Distillation Path {self.dataset_path}")

        self.current_dict_name = "DISTILLATION_FEATURE_ONLY_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dict_path = self.model_path / "runs" / self.current_dict_name
        run_dict_path.mkdir(parents=True, exist_ok=True)


        use_soft_distillation = True

        T = 2.0
        alpha = 0.05
        beta = 0.05

        soft_loss_function = nn.KLDivLoss(reduction="batchmean")

        for i in range(1, k_fold_value + 1):
            self.default_yolo_model = utils.load_yolo(self.model_path)
            self.yolo_model = self.default_yolo_model.model
            self.yolo_model.train()

            number_of_properties = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(
                number_of_properties,
                classes_count
            )

            self.yolo_model.to("cuda")

            teacher_weights = self.model_path / "model_utk_age_gender_4.23_97.69.pth"

            teacher_model = MiVOLOTrainer(
                weights_path=teacher_weights,
                classes_count=classes_count
            )

            teacher_model.to("cuda")
            teacher_model.eval()

            for param in teacher_model.parameters():
                param.requires_grad = False

            teacher_extractor = EmbeddingExtractor(
                model=teacher_model.mivolo,
                layer_name="norm"
            )

            student_layer_name = str(len(self.yolo_model.model) - 2)

            student_extractor = EmbeddingExtractor(
                model=self.yolo_model.model,
                layer_name=student_layer_name
            )

            fold_training_path = self.dataset_path / f"fold_{i}" / "train"

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            fold_dataset = datasets.ImageFolder(
                fold_training_path,
                transform=transform
            )

            loader = DataLoader(
                fold_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_data_loader_worker
            )


            print("Führe Dummy-Pass zur Feature-Analyse aus")

            with torch.no_grad():
                dummy_input = torch.rand(1, 3, 224, 224).to("cuda")

                _ = self.yolo_model(dummy_input)
                _ = teacher_model(dummy_input)

                dummy_student_features = pool_embedding(
                    student_extractor.get_embedding()
                )

                dummy_teacher_features = pool_embedding(
                    teacher_extractor.get_embedding()
                )

                student_dim = dummy_student_features.shape[1]
                teacher_dim = dummy_teacher_features.shape[1]

            feature_adapter = nn.Linear(
                student_dim,
                teacher_dim
            ).to("cuda")

            optimizer = torch.optim.AdamW(
                [
                    {"params": self.yolo_model.parameters()},
                    {"params": feature_adapter.parameters()}
                ],
                lr=self.learning_rate
            )

            print("Feature-Distillation Setup:")
            print(f"Student Feature Dim: {student_dim}")
            print(f"Teacher Feature Dim: {teacher_dim}")
            print(f"Adapter: {student_dim} -> {teacher_dim}")
            print(f"Soft-KD aktiv: {use_soft_distillation}")
            print(f"alpha = {alpha}, beta = {beta}")
            print()

            for epoch in range(epochs):
                epoch_total_loss = 0.0
                epoch_hard_loss = 0.0
                epoch_soft_loss = 0.0
                epoch_feature_loss = 0.0

                for inputs, labels in loader:
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")

                    optimizer.zero_grad()

                    student_predictions = self.yolo_model(inputs)

                    if isinstance(student_predictions, (list, tuple)):
                        student_predictions = student_predictions[0]

                    with torch.no_grad():
                        teacher_logits = teacher_model(inputs)

                    hard_loss = self.loss_function(
                        student_predictions,
                        labels
                    )

                    if use_soft_distillation:
                        soft_student = F.log_softmax(
                            student_predictions / T,
                            dim=-1
                        )

                        soft_teacher = F.softmax(
                            teacher_logits / T,
                            dim=-1
                        )

                        soft_loss = soft_loss_function(
                            soft_student,
                            soft_teacher
                        ) * (T ** 2)
                    else:
                        soft_loss = torch.tensor(
                            0.0,
                            device=inputs.device
                        )

                    student_features = student_extractor.get_embedding()
                    teacher_features = teacher_extractor.get_embedding()

                    student_features = pool_embedding(student_features)
                    teacher_features = pool_embedding(teacher_features).detach()

                    adapted_student_features = feature_adapter(
                        student_features
                    )

                    feature_loss = 1.0 - F.cosine_similarity(
                        adapted_student_features,
                        teacher_features,
                        dim=1
                    ).mean()

                    if use_soft_distillation:
                        loss = (
                                (1.0 - alpha) * hard_loss
                                + alpha * soft_loss
                                + beta * feature_loss
                        )
                    else:
                        loss = hard_loss + beta * feature_loss

                    loss.backward()
                    optimizer.step()

                    epoch_total_loss += loss.item()
                    epoch_hard_loss += hard_loss.item()
                    epoch_soft_loss += soft_loss.item()
                    epoch_feature_loss += feature_loss.item()

                num_batches = len(loader)

                print(
                    f"Feature Distillation Fold {i} Epoch {epoch} | "
                    f"Total: {epoch_total_loss / num_batches:.4f} | "
                    f"Hard: {epoch_hard_loss / num_batches:.4f} | "
                    f"Soft: {epoch_soft_loss / num_batches:.4f} | "
                    f"Feature: {epoch_feature_loss / num_batches:.4f}"
                )

                self.logger.info(
                    f"Feature Distillation Fold {i} Epoch {epoch} | "
                    f"Total: {epoch_total_loss / num_batches:.4f} | "
                    f"Hard: {epoch_hard_loss / num_batches:.4f} | "
                    f"Soft: {epoch_soft_loss / num_batches:.4f} | "
                    f"Feature: {epoch_feature_loss / num_batches:.4f}"
                )

            torch.save(
                self.yolo_model.state_dict(),
                run_dict_path / f"yolo26n-cls_fold{i}.pt"
            )

            teacher_extractor.remove()
            student_extractor.remove()


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


    #model.train_default_yolo(split_type.KFOLD, class_type.Group, dataset_type.DEFAULT, epochs=model.num_epochs,
    # k_fold_value=1,classes_count=16)
    # model.evaluate(model,16,class_type.Group,1,"Unscaled Dataset")
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
    model.train_distillation_yolo(split=split_type.KFOLD,class_type=class_type.Group,dataset_type=dataset_type.SCALED,epochs=model.num_epochs,k_fold_value=1,classes_count=16)
    model.evaluate(model, class_count=16, class_type=class_type.Group, k_fold_value=1, name="Distilled Model - Scaled")

