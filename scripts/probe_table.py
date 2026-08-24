"""Inspect raw table detection on the Inversis sample."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz

document = fitz.open("input_pdf/inversis_ssi_document.pdf")
page = document.load_page(0)
finder = page.find_tables()
print("tables found:", len(finder.tables))
for table_index, table in enumerate(finder.tables):
    rows = table.extract()
    print(f"\n=== table {table_index}: {len(rows)} rows x {max(len(r) for r in rows)} cols ===")
    for row_index, row in enumerate(rows[:6]):
        cleaned = [(cell or "").replace("\n", "\n")[:26] for cell in row]
        print(f" r{row_index}: {cleaned}")
    print(f" ... total rows {len(rows)}")
document.close()
