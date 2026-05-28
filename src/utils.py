import yaml
from pathlib import Path

def load_config():
    config_file = Path(__file__).parent.parent /"config"/ "config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)