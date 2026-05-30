import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import audio as A  # noqa: E402


def test_feature_extraction_shape():
    feat = A.extract_features(A.synthetic_waveform(1))
    assert feat.shape == (A.FEAT_DIM,)


def test_dataset_and_model():
    X, y = A.synthetic_dataset(per_class=10)
    assert X.shape[1] == A.FEAT_DIM and len(set(y.tolist())) == len(A.EMOTIONS)
    out = A.EmotionMLP()(X[:4])
    assert out.shape == (4, len(A.EMOTIONS))


def test_trains_on_separable_classes():
    from _shared.torch_utils import accuracy, train_classifier

    ld = A.loader(per_class=40)
    model = A.EmotionMLP()
    train_classifier(model, ld, epochs=12, device=torch.device("cpu"))
    # Synthetic emotions are spectrally separable → should learn well above chance (1/6).
    assert accuracy(model, ld, device=torch.device("cpu")) > 0.4


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"epochs": 3}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("demo", state), list)
