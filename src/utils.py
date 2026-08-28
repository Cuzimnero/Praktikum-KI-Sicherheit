import yaml
from pathlib import Path

from ultralytics import YOLO


def load_config():
    config_file = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yolo(model_path):
    """Lädt yolo Model herunter vom Webservice"""
    return YOLO(model_path / "yolo26n-cls.pt")


def get_dataset_path(dataset_type, class_type, split_type, classes_count:int, main_path):
    """Bestimmt Datensatzpfad Pfad anhand von Flags
    Parameter
    ----------
    dataset_type : dataset_type enum
        Daten Verarbeitungstype skalierte Trainingdaten Klassen, oder unskalierte Daten
    class_type : class_type enum
        Datensatzpfad Klassen Aufteilung in Gruppen oder ohne Gruppen
    split_type : split_type enum
        Datensatzpfad split Ansatz Standard K-Fold
    classes_count : int
    main_path : Path
        Hauptpfad zu Projekt Ordner
    Returns
    -------
   Path
       Gefundener Datensatzpfad

    Raises
    ------
    TypeError
        Falls split_type nicht existiert
    """
    dataset_path = None
    if split_type is split_type.KFOLD:
        if class_type is class_type.Group:
            if dataset_type is dataset_type.SCALED:
                dataset_path = main_path / "data" / "processed" / "scaled" / "k_fold" / "groups" / f"group_size_{classes_count}"

            else:
                dataset_path = main_path / "data" / "processed" / "k_fold" / "groups" / f"group_size_{classes_count}"

        elif class_type is class_type.default:
            if dataset_type is dataset_type.SCALED:
                dataset_path = main_path / "data" / "processed" / "scaled" / "k_fold" / "default"
            else:
                dataset_path = main_path / "data" / "processed" / "k_fold" / "default"
    else:
        raise TypeError("Invalid split type")
    return dataset_path
