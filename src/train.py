import json
import logging
from datetime import datetime
from pathlib import Path
from src.split_type import split_type,class_type,dataset_type,train_type
import src.utils as utils
import torch.nn as nn
from src.evaluation import evaluator
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.embedding_extractor import EmbeddingExtractor
from src.miovolo_trainer import MiVOLOTrainer
import torch.nn.functional as F


class model_trainer:
    def __init__(self):
        config=utils.load_config()

        self.main_path = Path(__file__).parent.parent
        self.model_path = self.main_path / config["paths"]["model_path"]
        self.logging_path = self.main_path / config["paths"]["logging_path"]
        self.teacher_weights_path = self.main_path / config["paths"]["teacher_weights_path"]

        self.num_yolo_epochs = config["train"]["yolo_num_epochs"]
        self.num_destillation_epochs=config["train"]["destillation_num_epochs"]
        self.classes_count = config["train"]["classes_count"]
        self.batch_size = config["train"]["batch_size"]
        self.learning_rate = config["train"]["learning_rate"]
        self.num_data_loader_worker = int(config["train"]["num_data_loader_worker"])
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.alpha=config["train"]["alpha"]
        self.beta=config["train"]["beta"]
        self.temperature=config["train"]["temperature"]
        self.sigma=config["train"]["sigma"]

        level_mapping = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        logging_mode_str = config["logging"].get("logging_mode")
        logging_mode = level_mapping.get(logging_mode_str, logging.INFO)
        self.log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        self.logging_path.mkdir(exist_ok=True, parents=True)
        log_file = self.logging_path / self.log_filename
        logging.basicConfig(
            filename=str(log_file),
            filemode='a',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging_mode,
            force=True
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger started")

        self.dataset_path = None
        self.current_dict_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.loss_function = nn.CrossEntropyLoss()

    def train_default_yolo(self, split_type: split_type, class_type: class_type,dataset_type:dataset_type, epochs: int, k_fold_value: int,
                        classes_count:int):
        self.dataset_path=utils.get_dataset_path(dataset_type,class_type,split_type,classes_count,self.main_path)
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

            self.logger.debug("Class to idx:")
            self.logger.debug(fold_dataset.class_to_idx)


            loader = DataLoader(fold_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_data_loader_worker)


            actual_classes_count = len(fold_dataset.classes)
            in_features = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(in_features, actual_classes_count)


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
            split_type: split_type,
            class_type: class_type,
            dataset_type: dataset_type,
            epochs: int,
            k_fold_value: int,
            classes_count: int,
            use_soft_distillation:bool

    ):
        self.dataset_path = utils.get_dataset_path(dataset_type, class_type, split_type, classes_count, self.main_path)

        self.logger.info(f"Using Distillation Path {self.dataset_path}")

        run_dict_path = self.model_path / "runs" / self.current_dict_name
        run_dict_path.mkdir(parents=True, exist_ok=True)

        soft_loss_function = nn.KLDivLoss(reduction="batchmean")

        for i in range(1, k_fold_value + 1):
            self.default_yolo_model = utils.load_yolo(self.model_path)
            self.yolo_model = self.default_yolo_model.model
            self.yolo_model.train()

            fold_training_path = self.dataset_path / f"fold_{i}" / "train"

            transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

            fold_dataset = datasets.ImageFolder(fold_training_path,transform=transform)

            actual_classes_count = len(fold_dataset.classes)

            self.logger.debug(f"Gewünschte Klassenanzahl: {classes_count}")
            self.logger.debug(f"Tatsächlich gefundene Klassenanzahl: {actual_classes_count}")

            if actual_classes_count != classes_count:
                self.logger.warning(f"Achtung: Es wurde classes_count={classes_count} angegeben, "f"aber ImageFolder hat nur {actual_classes_count} Klassen gefunden.")
                self.logger.warning("Für diesen Lauf wird actual_classes_count benutzt.")

            loader = DataLoader(fold_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_data_loader_worker)

            number_of_properties = self.yolo_model.model[-1].linear.in_features
            self.yolo_model.model[-1].linear = nn.Linear(number_of_properties,actual_classes_count)

            self.yolo_model.to("cuda")

            for param in self.yolo_model.parameters():
                param.requires_grad = True

            teacher_model = MiVOLOTrainer(weights_path=self.teacher_weights_path,class_type=class_type,dataset_path=self.dataset_path,sigma= self.sigma,logger=self.logger,class_names = fold_dataset.classes)

            teacher_model.to("cuda")
            teacher_model.eval()

            for param in teacher_model.parameters():param.requires_grad = False

            self.teacher_extractor = EmbeddingExtractor(model=teacher_model.mivolo,layer_name="norm")

            student_layer_name = str(len(self.yolo_model.model) - 2)

            student_extractor = EmbeddingExtractor(
                model=self.yolo_model.model,
                layer_name=student_layer_name
            )

            self.logger.debug("Künstliches Bild wird geschickt")

            with torch.no_grad():
                dummy_input = torch.rand(1, 3, 224, 224).to("cuda")

                _ = self.yolo_model(dummy_input)
                _ = teacher_model(dummy_input)

                dummy_student_features = self.pool_embedding(student_extractor.get_embedding())

                dummy_teacher_features = self.pool_embedding(self.teacher_extractor.get_embedding())

                student_dim = dummy_student_features.shape[1]
                teacher_dim = dummy_teacher_features.shape[1]

            feature_adapter = nn.Linear(student_dim,teacher_dim).to("cuda")

            optimizer = torch.optim.AdamW([{"params": self.yolo_model.parameters(), "lr": self.learning_rate},{"params": feature_adapter.parameters(), "lr": self.learning_rate}])

            self.logger.debug("Feature-Distillation Setup:")
            self.logger.debug(f"Student Feature Dim: {student_dim}")
            self.logger.debug(f"Teacher Feature Dim: {teacher_dim}")
            self.logger.debug(f"Adapter: {student_dim} -> {teacher_dim}")
            self.logger.debug(f"Soft-KD aktiv: {use_soft_distillation}")
            self.logger.debug(f"alpha = {self.alpha}, beta = {self.beta}")

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

                    hard_loss = self.loss_function(student_predictions,labels)

                    label_ages = teacher_model.class_centers[labels]
                    valid_teacher_interval = (label_ages >= 21) & (label_ages <= 60)


                    if use_soft_distillation and valid_teacher_interval.any():
                        soft_student = F.log_softmax(student_predictions[valid_teacher_interval] / self.temperature,dim=-1)
                        soft_teacher = F.softmax(teacher_logits[valid_teacher_interval] / self.temperature,dim=-1)

                        soft_loss = soft_loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
                    else:
                        soft_loss = torch.tensor(0.0, device=inputs.device)


                    student_features = student_extractor.get_embedding()
                    teacher_features = self.teacher_extractor.get_embedding()

                    student_features = self.pool_embedding(student_features)
                    teacher_features = self.pool_embedding(teacher_features).detach()

                    adapted_student_features = feature_adapter(student_features)

                    feature_loss = 1.0 - F.cosine_similarity(adapted_student_features,teacher_features,dim=1).mean()

                    if use_soft_distillation:
                        loss = ((1.0 - self.alpha) * hard_loss+ self.alpha * soft_loss+ self.beta * feature_loss)
                    else:
                        loss = hard_loss + self.beta * feature_loss

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

            torch.save(self.yolo_model.state_dict(),run_dict_path / f"yolo26n-cls_distillation_fold{i}.pt")

            self.teacher_extractor.remove()
            student_extractor.remove()


    def evaluate(self,model:model_trainer,class_count:int,class_type:class_type,k_fold_value:int,name:str,train_type:train_type):
        eval = evaluator(model.current_dict_name, model.model_path, model.batch_size, model.dataset_path,
                         class_count, model.logger, model.main_path, model.num_data_loader_worker,train_type)

        print(f"Average default training accuracy {eval.val_default_yolo(k_fold_value)}")

        if class_type is class_type.Group:
            with open(model.dataset_path / "group-list.json", "r", encoding="utf-8") as file:
                groups = json.load(file)
            group_list = [group["name"] for group in groups.values()]
        else:
            group_list = []
        eval.show_confusion_matrix(group_list, name)
        eval.safe_confusion_matrix()


    def pool_embedding(self, x):
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

