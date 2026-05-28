from enum import Enum
class split_type(Enum):
    KFOLD=1
    TRAIN_TEST=2

class class_type(Enum):
    Group=3
    default=4

class dataset_type(Enum):
    DEFAULT=1
    SCALED=2