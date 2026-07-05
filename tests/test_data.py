import yaml
from src.data import load_data


def test_load_data_works():
    cfg = yaml.safe_load(open("config/train.yaml", "r", encoding="utf-8"))
    bundle = load_data(cfg)
    assert bundle.X.shape[0] == bundle.y.shape[0]
    assert bundle.X.shape[0] > 0
