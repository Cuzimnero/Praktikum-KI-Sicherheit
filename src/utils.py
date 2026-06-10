import yaml
from pathlib import Path

from ultralytics import YOLO


def load_config():
    config_file = Path(__file__).parent.parent /"config"/ "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_yolo(model_path):
    return YOLO(model_path / "yolo26n-cls.pt")