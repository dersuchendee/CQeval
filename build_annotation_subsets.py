"""
Build 4 Human Annotation Subsets
----------------------------------
Creates 4 non-overlapping, random annotation subsets (~15 CQs each) for
inter-annotator coverage of:
  • source_deviation  — Y / N  (does the CQ introduce source inconsistency?)
  • framing_bias      — Y / N  (does the CQ frame the question with a bias?)

Gold standard labels (from human annotations in the source files) are stored
in gold_source_deviation and gold_framing_bias. These are only in the
*_with_gold.csv files — NOT in the files distributed to annotators.

source_consistency mapping:
  "Yes" (inconsistency present) → gold_source_deviation = Y
  "No"  (consistent)            → gold_source_deviation = N
  NaN / not annotated           → gold_source_deviation = "" (no gold available)

framing_bias mapping:
  "Yes" → gold_framing_bias = Y
  "No"  → gold_framing_bias = N
  NaN   → gold_framing_bias = "" (no gold available)

For IB/WHOW/CVN datasets there is no human annotation — gold columns will be empty.

Outputs per subset N:
  results/annotation_subset_{N}_annotator.csv   ← distribute to annotators
  results/annotation_subset_{N}_with_gold.csv   ← researcher's reference (contains gold)
"""

import re
import pathlib
import pandas as pd

RESULTS_DIR = BASE_DIR / "results"

EXCLUDE_PREFIXES = [
    "Persona\n\nOrtenz",
    "Successfully plan the restoration of an organ?",
    "The dashboard presents an overview",
]
_MULTI_CQ_BLOCK = re.compile(r"CQ\s*\d+\s*:.*CQ\s*\d+\s*:", re.DOTALL)

def excluded(s):   return any(str(s).startswith(p) for p in EXCLUDE_PREFIXES)
def is_real_cq(t): return not bool(_MULTI_CQ_BLOCK.search(str(t)))

def sc_to_gold(val):
    """source_consistency → gold_source_deviation (Y/N/"")"""
    s = str(val).strip() if pd.notna(val) else ""
    if s == "Yes": return "Y"
    if s == "No":  return "N"
    return ""

def fb_to_gold(val):
    """framing_bias column → gold_framing_bias (Y/N/"")"""
    s = str(val).strip() if pd.notna(val) else ""
    if s == "Yes": return "Y"
    if s == "No":  return "N"
    return ""


# ── Collect all CQs ──────────────────────────────────────────────────────────
records = []

# 1. OntoChat Polifonia (gold + OntoChat) — has source_consistency + framing_bias
df = pd.read_csv(
    str(BASE_DIR / "data" / "polifonia_ontochat_cqs.csv")
)
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src   = str(row["Source"]).strip()
    group = "Gold" if src == "gold" else "OntoChat"
    cq    = str(row.get("Question", "")).strip()
    if not is_real_cq(cq):
        continue
    records.append({
        "dataset":              "Polifonia",
        "group":                group,
        "scenario":             str(row["Scenario"]).strip(),
        "cq":                   cq,
        "gold_source_deviation": sc_to_gold(row.get("source_consistency")),
        "gold_framing_bias":     fb_to_gold(row.get("framing_bias")),
    })

# 2. NeoNGPT Polifonia — has source_consistency + framing_bias
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_polifonia_superfluous.csv"))
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src   = str(row["Source"]).strip()
    group = "NeoNGPT_GPT4o" if src == "GPT4o" else "NeoNGPT_Qwen3"
    cq    = str(row.get("Question", "")).strip()
    if not is_real_cq(cq):
        continue
    records.append({
        "dataset":              "Polifonia",
        "group":                group,
        "scenario":             str(row["Scenario"]).strip(),
        "cq":                   cq,
        "gold_source_deviation": sc_to_gold(row.get("source_consistency")),
        "gold_framing_bias":     fb_to_gold(row.get("framing_bias")),
    })

# 3. OntoChat IB/WHOW/CVN — no human annotation
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv"))
for _, row in df.iterrows():
    sys   = str(row.get("System", "")).strip()
    group = "Gold" if sys == "Gold_Standard_CQs" else "OntoChat"
    cq    = str(row.get("CQ", "")).strip()
    if not is_real_cq(cq):
        continue
    records.append({
        "dataset":              str(row.get("Project Name", "")).strip(),
        "group":                group,
        "scenario":             str(row["Scenario"]).strip(),
        "cq":                   cq,
        "gold_source_deviation": "",
        "gold_framing_bias":     "",
    })

# 4. NeoNGPT IB/WHOW/CVN — no human annotation
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv"))
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys   = str(row.get("System", "")).strip()
    group = "NeoNGPT_GPT4o" if sys == "GPT4o_CQs" else "NeoNGPT_Qwen3"
    cq    = str(row.get("CQ", "")).strip()
    if not is_real_cq(cq):
        continue
    records.append({
        "dataset":              str(row.get("Project Name", "")).strip(),
        "group":                group,
        "scenario":             str(row["Scenario"]).strip(),
        "cq":                   cq,
        "gold_source_deviation": "",
        "gold_framing_bias":     "",
    })

all_df = pd.DataFrame(records).drop_duplicates(subset=["cq"]).reset_index(drop=True)
print(f"Total unique CQs in pool: {len(all_df)}")
print(f"Polifonia rows with gold_source_deviation: "
      f"{(all_df['gold_source_deviation'] != '').sum()}")
print(f"Polifonia rows with gold_framing_bias:     "
      f"{(all_df['gold_framing_bias'] != '').sum()}")

# ── Stratified sampling into 4 non-overlapping subsets ───────────────────────
# ~15 CQs per subset: Polifonia 4, IB 4, CVN 4, WHOW 3
DATASET_QUOTA = {
    "Polifonia":            4,
    "IntelligentBathrooms": 4,
    "CVN":                  4,
    "WHOW":                 3,
}
N_SUBSETS = 4
SEED      = 99

assigned = []
for dataset, quota in DATASET_QUOTA.items():
    sub  = all_df[all_df["dataset"] == dataset].sample(frac=1, random_state=SEED).reset_index(drop=True)
    need = quota * N_SUBSETS
    if len(sub) < need:
        sub = pd.concat([sub] * (need // len(sub) + 1)).head(need).reset_index(drop=True)
    pool = sub.head(need).copy()
    pool["_subset"] = [i % N_SUBSETS + 1 for i in range(len(pool))]
    assigned.append(pool)

combined = pd.concat(assigned, ignore_index=True)

# ── Save per subset ──────────────────────────────────────────────────────────
ANNOTATOR_COLS = [
    "dataset",
    "group",
    "scenario",          # full scenario text
    "cq",
    # blank columns to fill in
    "source_deviation",  # Y = CQ introduces inconsistency with scenario; N = consistent
    "framing_bias",      # Y = CQ introduces a framing bias; N = neutral
    "notes",
]

GOLD_COLS = ANNOTATOR_COLS + [
    "gold_source_deviation",   # mapped from source_consistency (Polifonia only)
    "gold_framing_bias",       # mapped from framing_bias column (Polifonia only)
]

for s in range(1, N_SUBSETS + 1):
    sub_df = combined[combined["_subset"] == s].copy()
    sub_df = sub_df.sample(frac=1, random_state=SEED + s).reset_index(drop=True)
    for col in ["source_deviation", "framing_bias", "notes"]:
        sub_df[col] = ""

    ann_path  = RESULTS_DIR / f"annotation_subset_{s}_annotator.csv"
    gold_path = RESULTS_DIR / f"annotation_subset_{s}_with_gold.csv"

    sub_df[ANNOTATOR_COLS].to_csv(ann_path,  index=False)
    sub_df[GOLD_COLS].to_csv(gold_path, index=False)

    print(f"\n── Subset {s} ({len(sub_df)} CQs) ──")
    print(f"  annotator  → {ann_path}")
    print(f"  with gold  → {gold_path}")
    print(sub_df.groupby(["dataset", "group"]).size().rename("n").to_string())
    has_gold = (sub_df["gold_source_deviation"] != "").sum()
    print(f"  rows with gold labels: {has_gold}/{len(sub_df)}")

# ── Overlap check ─────────────────────────────────────────────────────────────
all_cqs = []
for s in range(1, N_SUBSETS + 1):
    df_s = pd.read_csv(RESULTS_DIR / f"annotation_subset_{s}_annotator.csv")
    df_s["_subset"] = s
    all_cqs.append(df_s)
all_cqs_df = pd.concat(all_cqs)
dupes = all_cqs_df.groupby("cq")["_subset"].nunique()
overlapping = dupes[dupes > 1]
if overlapping.empty:
    print("\n✓ No overlapping CQs across subsets.")
else:
    print(f"\n⚠ {len(overlapping)} CQs appear in multiple subsets!")
