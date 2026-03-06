# src/save_results.py
import json
from pathlib import Path
from typing import Dict, Any


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """
    Append one experiment record to a JSONL file.

    Each line = one JSON object.
    """
    p = Path(path)

    # create outputs directory if it doesn't exist
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")