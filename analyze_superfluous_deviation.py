"""
Superfluous Deviation Analysis
-------------------------------
Measures per-scenario prevalence (A) and intensity (B) of superfluous
elements for each CQ generation group, then reports median + IQR as
the normal range for each group.

Deviation types (Expansion / Specification / Implicit Assumption /
Restriction) are included where annotated (OntoChat Polifonia file).

Groups
------
  Gold           human-written CQs                       (no LLM)
  OntoChat       GPT-4o, single-step                     closed LLM
  NeoNGPT_GPT4o  GPT-4o, two-step NeOn                  closed LLM
  NeoNGPT_Qwen3  Ollama Qwen3-14b, two-step NeOn        open LLM

Outputs (results/)
------------------
  superfluous_deviation_per_scenario.csv   per (scenario, group) stats
  superfluous_deviation_summary.csv        median + IQR per group
  superfluous_deviation_types.csv          deviation-type breakdown
"""

import re
import pathlib
import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ONTOCHAT_POLIFONIA    = pathlib.Path(
    str(BASE_DIR / "data" / "polifonia_ontochat_cqs.csv")
)
NEONGPT_POLIFONIA     = pathlib.Path(
    str(BASE_DIR / "data" / "annotated_neongpt_polifonia_superfluous.csv")
)
NEONGPT_IB_WHOW_CVN  = pathlib.Path(
    str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv")
)
ONTOCHAT_IB_WHOW_CVN = pathlib.Path(
    str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv")
)
RESULTS_DIR = BASE_DIR / "results"

# ── Polifonia scenarios to exclude ────────────────────────────────────────────
EXCLUDE_PREFIXES = [
    "Persona\n\nOrtenz",
    "Successfully plan the restoration of an organ?",
    "The dashboard presents an overview",
]

# ── helpers ────────────────────────────────────────────────────────────────────
def _excluded(scenario: str) -> bool:
    return any(str(scenario).startswith(p) for p in EXCLUDE_PREFIXES)


_MULTI_CQ_BLOCK = re.compile(r"CQ\s*\d+\s*:.*CQ\s*\d+\s*:", re.DOTALL)

def _is_real_cq(cq_text: str) -> bool:
    """Return False for cells where the model dumped a numbered list of CQs."""
    return not bool(_MULTI_CQ_BLOCK.search(str(cq_text)))


def parse_superfluous(val):
    """
    Returns (count: int | None, is_valid: bool).

    "0" / "No"    → (0, True)
    "N: ..."       → (N, True)
    "N (note)"     → (N, True)
    loose terms    → (n_terms, True)   e.g. "disposition//point in time"
    api_error/NaN  → (None, False)
    """
    if pd.isna(val):
        return None, False
    s = str(val).strip()
    if s in ("", "api_error", "?", "nan"):
        return None, False
    if s in ("0", "No"):
        return 0, True
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1)), True
    # no leading number — count slash/comma-separated terms
    terms = [t.strip() for t in re.split(r"[,/]+", s) if t.strip()]
    return (len(terms), True) if terms else (None, False)


# ── load & unify all sources ───────────────────────────────────────────────────
records = []

# 1. OntoChat Polifonia  (gold + OntoChat, has deviation_types)
df = pd.read_csv(ONTOCHAT_POLIFONIA)
df = df[~df["Scenario"].apply(_excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group   = "Gold"     if src == "gold"    else "OntoChat"
    llm_type = "human"   if src == "gold"    else "closed"
    cnt, valid = parse_superfluous(row.get("superfluous_elements"))
    dev_type = (
        str(row["deviation_types"]).strip()
        if pd.notna(row.get("deviation_types")) and str(row.get("deviation_types")).strip()
        else None
    )
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                str(row.get("Question", "")).strip(),
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "Polifonia",
        "superfluous_count": cnt,
        "is_valid":          valid and _is_real_cq(row.get("Question", "")),
        "deviation_types":   dev_type,
    })

# 2. NeoNGPT Polifonia  (GPT4o + Qwen3)
df = pd.read_csv(NEONGPT_POLIFONIA)
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    if src == "GPT4o":
        group, llm_type = "NeoNGPT_GPT4o", "closed"
    else:
        group, llm_type = "NeoNGPT_Qwen3", "open"
    cnt, valid = parse_superfluous(row.get("superfluous_elements"))
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                str(row.get("Question", "")).strip(),
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "Polifonia",
        "superfluous_count": cnt,
        "is_valid":          valid and _is_real_cq(row.get("Question", "")),
        "deviation_types":   None,
    })

# 3. OntoChat IB/WHOW/CVN  (gold — taken here only to avoid double-counting)
df = pd.read_csv(ONTOCHAT_IB_WHOW_CVN)
for _, row in df.iterrows():
    sys = str(row["System"]).strip()
    if sys == "Gold_Standard_CQs":
        group, llm_type = "Gold", "human"
    else:
        group, llm_type = "OntoChat", "closed"
    cnt, valid = parse_superfluous(row.get("superfluous_elements"))
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                str(row.get("CQ", "")).strip(),
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "IB_WHOW_CVN",
        "superfluous_count": cnt,
        "is_valid":          valid and _is_real_cq(row.get("CQ", "")),
        "deviation_types":   None,
    })

# 4. NeoNGPT IB/WHOW/CVN  (GPT4o + Qwen3; gold excluded — already loaded above)
df = pd.read_csv(NEONGPT_IB_WHOW_CVN)
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys = str(row["System"]).strip()
    if sys == "GPT4o_CQs":
        group, llm_type = "NeoNGPT_GPT4o", "closed"
    else:
        group, llm_type = "NeoNGPT_Qwen3", "open"
    cnt, valid = parse_superfluous(row.get("superfluous_elements"))
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                str(row.get("CQ", "")).strip(),
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "IB_WHOW_CVN",
        "superfluous_count": cnt,
        "is_valid":          valid and _is_real_cq(row.get("CQ", "")),
        "deviation_types":   None,
    })

all_df = pd.DataFrame(records)

print(f"Total records loaded: {len(all_df)}")
print(all_df.groupby(["group", "llm_type", "dataset"])["is_valid"].agg(["sum", "count"]).rename(columns={"sum": "valid", "count": "total"}))

# ── per-scenario measures ──────────────────────────────────────────────────────
valid_df = all_df[all_df["is_valid"]].copy()

scenario_stats = (
    valid_df
    .groupby(["scenario", "group", "llm_type", "dataset"], sort=False)
    .agg(
        n_valid       = ("superfluous_count", "count"),
        n_superfluous = ("superfluous_count", lambda x: (x > 0).sum()),
        sum_count     = ("superfluous_count", "sum"),
    )
    .reset_index()
)
scenario_stats["A_prevalence"] = (
    scenario_stats["n_superfluous"] / scenario_stats["n_valid"]
)
scenario_stats["B_intensity"] = (
    scenario_stats["sum_count"] / scenario_stats["n_valid"]
)
# short label for readability
scenario_stats["scenario_short"] = scenario_stats["scenario"].str[:60]

# ── summary: median + IQR per group ───────────────────────────────────────────
summary_rows = []
for (group, llm_type), sub in scenario_stats.groupby(["group", "llm_type"]):
    for measure_label, col in [("A_prevalence", "A"), ("B_intensity", "B")]:
        s = sub[measure_label].dropna()
        if len(s) == 0:
            continue
        summary_rows.append({
            "group":       group,
            "llm_type":    llm_type,
            "measure":     col,
            "n_scenarios": len(s),
            "median":      round(s.median(), 3),
            "Q1":          round(s.quantile(0.25), 3),
            "Q3":          round(s.quantile(0.75), 3),
            "IQR":         round(s.quantile(0.75) - s.quantile(0.25), 3),
            "min":         round(s.min(), 3),
            "max":         round(s.max(), 3),
        })

summary_df = pd.DataFrame(summary_rows)

# ── deviation types (where annotated) ─────────────────────────────────────────
dev_df = all_df[
    all_df["deviation_types"].notna() & (all_df["deviation_types"] != "")
].copy()

type_rows = []
if not dev_df.empty:
    for (group, dataset), sub in dev_df.groupby(["group", "dataset"]):
        total = len(sub)
        for dtype, cnt in sub["deviation_types"].value_counts().items():
            type_rows.append({
                "group":      group,
                "dataset":    dataset,
                "dev_type":   dtype,
                "count":      cnt,
                "total_with_deviation": total,
                "proportion": round(cnt / total, 3),
            })

type_df = pd.DataFrame(type_rows) if type_rows else pd.DataFrame()

# ── save ───────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(exist_ok=True)

out_scenario = RESULTS_DIR / "superfluous_deviation_per_scenario.csv"
out_summary  = RESULTS_DIR / "superfluous_deviation_summary.csv"
out_types    = RESULTS_DIR / "superfluous_deviation_types.csv"

scenario_stats.to_csv(out_scenario, index=False)
summary_df.to_csv(out_summary, index=False)
if not type_df.empty:
    type_df.to_csv(out_types, index=False)

# ── print results ──────────────────────────────────────────────────────────────
ORDER = ["Gold", "OntoChat", "NeoNGPT_GPT4o", "NeoNGPT_Qwen3"]
summary_df["_ord"] = summary_df["group"].apply(
    lambda g: ORDER.index(g) if g in ORDER else 99
)
summary_df = summary_df.sort_values(["_ord", "measure"]).drop(columns="_ord")

print("\n\n╔══════════════════════════════════════════════════════════════╗")
print("║   SUPERFLUOUS DEVIATION — NORMAL RANGE (median + IQR)        ║")
print("╚══════════════════════════════════════════════════════════════╝\n")

for measure_label, mkey in [
    ("A  — Prevalence  (proportion of CQs with ≥1 superfluous element)", "A"),
    ("B  — Intensity   (mean superfluous elements per CQ)",               "B"),
]:
    print(f"── {measure_label} ──\n")
    sub = summary_df[summary_df["measure"] == mkey][[
        "group", "llm_type", "n_scenarios", "median", "Q1", "Q3", "IQR", "min", "max"
    ]]
    print(sub.to_string(index=False))
    print()

if not type_df.empty:
    print("\n── Deviation types (where annotated) ──\n")
    print(type_df.sort_values(["group", "proportion"], ascending=[True, False]).to_string(index=False))

print(f"\nSaved:\n  {out_scenario}\n  {out_summary}")
if not type_df.empty:
    print(f"  {out_types}")
