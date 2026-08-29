import logging

import torch
from sympy.abc import sigma
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.miovolo_trainer import MiVOLOTrainer
from src.split_type import class_type


def evaluate_teacher(weights_path, data_path, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)

    print("Klassen:", dataset.classes)
    print("Anzahl Klassen:", len(dataset.classes))
    print("Anzahl Bilder:", len(dataset))

    teacher = MiVOLOTrainer(
        weights_path=weights_path,
        class_type=class_type.default,
        dataset_path=data_path,
        sigma=sigma,
        logger=logging.getLogger(),
        class_names=dataset.classes

    )

    teacher.to(device)
    teacher.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    class_centers = teacher.class_centers.to(device)

    all_correct = 0
    all_total = 0
    all_mae = 0.0
    all_cs1 = 0

    mid_correct = 0
    mid_total = 0
    mid_mae = 0.0
    mid_cs1 = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = teacher(inputs)
            preds = outputs.argmax(dim=1)

            label_ages = class_centers[labels]
            pred_ages = class_centers[preds]
            age_errors = torch.abs(pred_ages - label_ages)

            all_total += labels.size(0)
            all_correct += (preds == labels).sum().item()
            all_mae += age_errors.sum().item()
            all_cs1 += (age_errors <= 1).sum().item()


            mask = (label_ages >= 20) & (label_ages <= 70)

            if mask.any():
                mid_labels = labels[mask]
                mid_preds = preds[mask]
                mid_errors = age_errors[mask]

                mid_total += mid_labels.size(0)
                mid_correct += (mid_preds == mid_labels).sum().item()
                mid_mae += mid_errors.sum().item()
                mid_cs1 += (mid_errors <= 1).sum().item()

    print("-" * 40)
    print("ALLE KLASSEN")
    print(f"Accuracy: {100 * all_correct / all_total:.2f}%")
    print(f"MAE:      {all_mae / all_total:.4f}")
    print(f"CS ±1:    {100 * all_cs1 / all_total:.2f}%")

    print("-" * 40)
    print("NUR MITTLERE ALTERSBEREICHE 20-70")

    if mid_total > 0:
        print(f"Samples:  {mid_total}")
        print(f"Accuracy: {100 * mid_correct / mid_total:.2f}%")
        print(f"MAE:      {mid_mae / mid_total:.4f}")
        print(f"CS ±1:    {100 * mid_cs1 / mid_total:.2f}%")
    else:
        print("Keine Samples für mittlere Altersbereiche gefunden.")

    print("-" * 40)


if __name__ == "__main__":
    W_PATH = r"C:\Users\HazimZ5\PycharmProjects\Praktikum-KI-Sicherheit\models\MiVOLO\checkpoints\model_age_utk_4.23.pth"

    D_PATH = r"C:\Users\HazimZ5\PycharmProjects\Praktikum-KI-Sicherheit\data\processed\k_fold\default\fold_2\val"

    evaluate_teacher(
        weights_path=W_PATH,
        data_path=D_PATH,
        batch_size=64
    )