import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import MiVOLOTrainer


def evaluate_teacher(weights_path, data_path, classes_count=16, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = MiVOLOTrainer(
        weights_path=weights_path,
        classes_count=classes_count
    )

    teacher.to(device)
    teacher.eval()

    # Wichtig:
    # Kein Normalize hier, wenn MiVOLOTrainer intern normalisiert:
    # x = (x - self.mean) / self.std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("Klassen:", dataset.classes)
    print("Anzahl Bilder:", len(dataset))

    all_correct = 0
    all_total = 0
    all_mae = 0
    all_cs1 = 0

    mid_correct = 0
    mid_total = 0
    mid_mae = 0
    mid_cs1 = 0

    # ImageFolder-Reihenfolge bei dir:
    # ['1-6', '13-18', '19-24', '25-30', '31-37', '38-43',
    #  '44-49', '50-55', '56-61', '62-68', '69-74',
    #  '7-12', '75-80', '81-86', '87-92', '93-110']
    #
    # Klassen 3-9 entsprechen:
    # 25-30 bis 62-68
    allowed_classes = {3, 4, 5, 6, 7, 8, 9}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = teacher(inputs)
            preds = outputs.argmax(dim=1)

            errors = torch.abs(preds - labels)

            all_total += labels.size(0)
            all_correct += (preds == labels).sum().item()
            all_mae += errors.sum().item()
            all_cs1 += (errors <= 1).sum().item()

            mask = torch.zeros_like(labels, dtype=torch.bool)

            for c in allowed_classes:
                mask |= labels == c

            if mask.any():
                mid_labels = labels[mask]
                mid_preds = preds[mask]
                mid_errors = torch.abs(mid_preds - mid_labels)

                mid_total += mid_labels.size(0)
                mid_correct += (mid_preds == mid_labels).sum().item()
                mid_mae += mid_errors.sum().item()
                mid_cs1 += (mid_errors <= 1).sum().item()

    print("-" * 40)
    print("ALLE KLASSEN")
    print(f"Accuracy: {100 * all_correct / all_total:.2f}%")
    print(f"MAE:      {all_mae / all_total:.4f}")
    print(f"CS ±1:    {100 * all_cs1 / all_total:.2f}")

    print("-" * 40)
    print("NUR MITTLERE KLASSEN")

    if mid_total > 0:
        print(f"Samples:  {mid_total}")
        print(f"Accuracy: {100 * mid_correct / mid_total:.2f}%")
        print(f"MAE:      {mid_mae / mid_total:.4f}")
        print(f"CS ±1:    {100 * mid_cs1 / mid_total:.2f}%")
    else:
        print("Keine Samples für mittlere Klassen gefunden.")

    print("-" * 40)


if __name__ == "__main__":
    W_PATH = r"C:\Users\HazimZ5\PycharmProjects\Praktikum-KI-Sicherheit\models\model_utk_age_gender_4.23_97.69.pth"

    D_PATH = r"C:\Users\HazimZ5\PycharmProjects\Praktikum-KI-Sicherheit\data\processed\scaled\k_fold\groups\group_size_16\fold_1\val"

    evaluate_teacher(
        weights_path=W_PATH,
        data_path=D_PATH,
        classes_count=16,
        batch_size=64
    )