"""
Disaggregates output_neongpt_polifonia_cqs.csv so that each
competency question gets its own row, with a Source column indicating
whether it is a gold standard CQ or which model generated it.

Output: disaggregate_neongpt_polifonia_cqs.csv
"""

import pandas as pd
import pathlib

INPUT  = pathlib.Path(str(BASE_DIR / "output_neongpt_polifonia_cqs.csv"))
OUTPUT = pathlib.Path(str(BASE_DIR / "disaggregate_neongpt_polifonia_cqs.csv"))

SOURCE_MAP = {
    "Gold_Standard_CQs":          "gold",
    "GPT4o_CQs":                  "GPT4o",
    "Ollama_qwen3_14b_CQs":       "Ollama_qwen3_14b",
}

df = pd.read_csv(INPUT)

rows = []
for _, agg_row in df.iterrows():
    scenario = agg_row["Scenario"]

    for col, source_label in SOURCE_MAP.items():
        cell = str(agg_row.get(col, "")).strip()
        if not cell or cell.lower() in ("nan", "") or cell.startswith("Error") or cell.startswith("Ollama error"):
            continue

        cqs = [cq.strip() for cq in cell.split(";") if cq.strip()]

        for cq in cqs:
            rows.append({
                "Scenario": scenario,
                "Question": cq,
                "Source":   source_label,
            })

out_df = pd.DataFrame(rows, columns=["Scenario", "Question", "Source"])
out_df.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"✓ Written {len(out_df)} rows to {OUTPUT}")
print(out_df["Source"].value_counts().to_string())
