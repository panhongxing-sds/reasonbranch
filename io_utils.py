"""Parquet/JSON persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.with_suffix(path.suffix + ".empty").write_text("[]")
        return

    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    except Exception:
        pass

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        return
    except Exception:
        pass

    json_path = path.with_suffix(".jsonl")
    with json_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_table(path: Path):
    if path.exists():
        try:
            import pandas as pd

            return pd.read_parquet(path)
        except Exception:
            pass
    json_path = path.with_suffix(".jsonl")
    if json_path.exists():
        import pandas as pd

        return pd.read_json(json_path, lines=True)
    raise FileNotFoundError(path)
