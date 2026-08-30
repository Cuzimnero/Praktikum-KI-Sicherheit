from pathlib import Path
import models.MiVOLO.mivolo.model.mivolo_model as mvm
import torch
import torch.nn as nn
import logging

class MiVOLOTrainer(nn.Module):
    def __init__(self, weights_path, class_type, dataset_path: Path, sigma,logger:logging.Logger,class_names=None):
        super().__init__()

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        self.logger = logger

        self.logger.debug("Checkpoint Informationen:")
        self.logger.debug(f"min_age = {checkpoint['min_age']}")
        self.logger.debug(f"max_age = {checkpoint['max_age']}")
        self.logger.debug(f"avg_age = {checkpoint['avg_age']}")
        self.logger.debug(f"no_gender = {checkpoint['no_gender']}")

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

        self.logger.debug(f"Geladene Parameter: {len(pretrained_dict)} / {len(model_dict)}")

        model_dict.update(pretrained_dict)
        self.mivolo.load_state_dict(model_dict, strict=False)

        for p in self.mivolo.parameters():
            p.requires_grad = False

        self.mivolo.eval()

        self.sigma = sigma

        if class_names is None:
            raise ValueError("class_names fehlt. Übergib fold_dataset.classes an MiVOLOTrainer.")

        class_centers = compute_class_centers_from_names(class_names)

        self.logger.debug("Class Centers passend zu ImageFolder:")
        for idx, (name, center) in enumerate(zip(class_names, class_centers.tolist())):
            self.logger.debug(f"{idx}: {name} -> {center}")

        self.register_buffer("class_centers", class_centers)



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

        diff = age.unsqueeze(1) - self.class_centers.unsqueeze(0)

        teacher_logits = -(diff ** 2) / (2 * self.sigma ** 2)

        return teacher_logits

def compute_class_centers_from_names(class_names):
    centers = []

    for name in class_names:
        name = str(name)

        if "-" in name:
            start, end = name.split("-")
            center = (float(start) + float(end)) / 2.0
        else:
            center = float(name)

        centers.append(center)

    return torch.tensor(centers, dtype=torch.float32)

