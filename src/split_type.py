from enum import Enum
#Hilfsenums für Codeübersichtlichkeit

class split_type(Enum):
    """Dataset split Ansatz"""
    KFOLD=1
    TRAIN_TEST=2

class class_type(Enum):
    """Dataset Klassen Aufteilung in Gruppen oder ohne Gruppen"""
    Group=3
    default=4

class dataset_type(Enum):
    """Daten Verarbeitungstype skalierte Trainingdaten Klassen, oder unskalierte Daten"""
    DEFAULT=1
    SCALED=2

class train_type(Enum):
    """Train-Methode Destillation oder normales Training"""
    default=1
    distillation=2