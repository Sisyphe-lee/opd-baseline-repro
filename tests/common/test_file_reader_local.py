import json

from trinity.buffer.reader.file_reader import _load_experience_dataset
from trinity.common.config import StorageConfig


def test_local_jsonl_experience_dataset_is_loaded_with_json_builder(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = _load_experience_dataset(StorageConfig(path=str(path), split="train"))
    assert len(dataset) == 1
    assert dataset[0]["messages"] == row["messages"]
