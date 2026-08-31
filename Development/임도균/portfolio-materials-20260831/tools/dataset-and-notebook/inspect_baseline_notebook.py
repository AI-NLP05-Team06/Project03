from __future__ import annotations

import json
import re
from pathlib import Path


path = Path(
    r"C:\Users\임도균\.codex\attachments"
    r"\e441a4a3-6d1b-45cb-9ade-687feba7a24d\pasted-text.txt"
)
notebook = json.loads(path.read_text(encoding="utf-8"))
print("cells", len(notebook["cells"]))
for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    first = next((line.strip() for line in source.splitlines() if line.strip()), "")
    definitions = re.findall(
        r"^(?:def|class)\s+([A-Za-z0-9_]+)", source, flags=re.MULTILINE
    )
    print(
        f"{index:02d} {cell['cell_type']:8} lines={len(source.splitlines()):3} "
        f"first={first[:100]} defs={definitions}"
    )
