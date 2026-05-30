import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import finetune as F  # noqa: E402


def test_format_example_contains_fields():
    s = F.format_example({"instruction": "Do X", "input": "abc", "output": "done"})
    assert "Do X" in s and "abc" in s and "done" in s
    assert "### Response:" in s


def test_build_dataset():
    ds = F.build_dataset(F.SAMPLE_DATA, eos="<eos>")
    assert len(ds) == len(F.SAMPLE_DATA)
    assert all(s.endswith("<eos>") for s in ds)


def test_handler_data_preview():
    from src import handler as H

    out = H.respond("data", {"config": {}})
    assert isinstance(out, list)
