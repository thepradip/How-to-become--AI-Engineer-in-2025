import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import face as Fc  # noqa: E402


def test_embedding_is_unit_norm():
    model = Fc.FaceNet()
    e = model.embed(torch.rand(3, 3, 32, 32))
    assert e.shape == (3, 64)
    assert torch.allclose(e.norm(dim=1), torch.ones(3), atol=1e-4)


def test_verify_returns_bool_and_similarity():
    model = Fc.FaceNet()
    same, sim = Fc.verify(model, torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    assert isinstance(same, bool) and -1.0 <= sim <= 1.0


def test_same_identity_more_similar_after_training():
    from _shared.torch_utils import train_classifier

    X, y = Fc.synthetic_identities(10, 30)
    model = Fc.FaceNet(n_ids=10)
    train_classifier(model, Fc.loader(10, 30), epochs=10, device=torch.device("cpu"))
    i0 = (y == 0).nonzero().flatten(); i1 = (y == 1).nonzero().flatten()
    _, same_sim = Fc.verify(model, X[i0[0]:i0[0] + 1], X[i0[1]:i0[1] + 1])
    _, diff_sim = Fc.verify(model, X[i0[0]:i0[0] + 1], X[i1[0]:i1[0] + 1])
    assert same_sim > diff_sim          # same person embeddings closer
