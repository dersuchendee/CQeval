"""
Build 4 Flag-Focused Annotation Subsets
-----------------------------------------
Draws exclusively from scenario×group combinations that are flagged ABOVE or
BELOW the IQR normal range for superfluous elements (SE) or eval terms (ET).

Flag columns included:
  se_A_flag / se_B_flag  — SE prevalence (A) and intensity (B) flags
  et_A_flag / et_B_flag  — eval-term prevalence and intensity flags
  Values: ABOVE | BELOW | ok

Subsets are balanced across flag direction (ABOVE / BELOW), dataset, and group.
Each subset ~15 CQs, completable in ~20 minutes.

Outputs per subset N:
  results/flag_subset_{N}_annotator.csv   ← distribute to annotators
  results/flag_subset_{N}_with_gold.csv   ← researcher reference (contains gold)
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
    s = str(val).strip() if pd.notna(val) else ""
    if s == "Yes": return "Y"
    if s == "No":  return "N"
    return ""

def fb_to_gold(val):
    s = str(val).strip() if pd.notna(val) else ""
    if s == "Yes": return "Y"
    if s == "No":  return "N"
    return ""

# ── Collect all CQs ──────────────────────────────────────────────────────────
records = []

# 1. OntoChat Polifonia
df = pd.read_csv(
    str(BASE_DIR / "data" / "polifonia_ontochat_cqs.csv")
)
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src   = str(row["Source"]).strip()
    group = "Gold" if src == "gold" else "OntoChat"
    cq    = str(row.get("Question", "")).strip()
    if not is_real_cq(cq): continue
    records.append({
        "dataset": "Polifonia", "group": group,
        "scenario": str(row["Scenario"]).strip(), "cq": cq,
        "gold_source_deviation": sc_to_gold(row.get("source_consistency")),
        "gold_framing_bias":     fb_to_gold(row.get("framing_bias")),
    })

# 2. NeoNGPT Polifonia
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_polifonia_superfluous.csv"))
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src   = str(row["Source"]).strip()
    group = "NeoNGPT_GPT4o" if src == "GPT4o" else "NeoNGPT_Qwen3"
    cq    = str(row.get("Question", "")).strip()
    if not is_real_cq(cq): continue
    records.append({
        "dataset": "Polifonia", "group": group,
        "scenario": str(row["Scenario"]).strip(), "cq": cq,
        "gold_source_deviation": sc_to_gold(row.get("source_consistency")),
        "gold_framing_bias":     fb_to_gold(row.get("framing_bias")),
    })

# 3. OntoChat IB/WHOW/CVN
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv"))
for _, row in df.iterrows():
    sys   = str(row.get("System", "")).strip()
    group = "Gold" if sys == "Gold_Standard_CQs" else "OntoChat"
    cq    = str(row.get("CQ", "")).strip()
    if not is_real_cq(cq): continue
    records.append({
        "dataset": str(row.get("Project Name", "")).strip(), "group": group,
        "scenario": str(row["Scenario"]).strip(), "cq": cq,
        "gold_source_deviation": "", "gold_framing_bias": "",
    })

# 4. NeoNGPT IB/WHOW/CVN
df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv"))
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys   = str(row.get("System", "")).strip()
    group = "NeoNGPT_GPT4o" if sys == "GPT4o_CQs" else "NeoNGPT_Qwen3"
    cq    = str(row.get("CQ", "")).strip()
    if not is_real_cq(cq): continue
    records.append({
        "dataset": str(row.get("Project Name", "")).strip(), "group": group,
        "scenario": str(row["Scenario"]).strip(), "cq": cq,
        "gold_source_deviation": "", "gold_framing_bias": "",
    })

all_df = pd.DataFrame(records).drop_duplicates(subset=["cq"]).reset_index(drop=True)

# ── Attach flag data ──────────────────────────────────────────────────────────
def norm_dataset(d):
    return "IB_WHOW_CVN" if d in ("IntelligentBathrooms", "WHOW", "CVN") else d

all_df["dataset_norm"] = all_df["dataset"].apply(norm_dataset)

se_flags = pd.read_csv(RESULTS_DIR / "superfluous_deviation_flagged.csv")
et_flags = pd.read_csv(RESULTS_DIR / "eval_terms_deviation_flagged.csv")

se_lookup = {
    (str(r["scenario"])[:60], str(r["group"]), str(r["dataset"])): (str(r["A_flag"]), str(r["B_flag"]))
    for _, r in se_flags.iterrows()
}
et_lookup = {
    (str(r["scenario"])[:60], str(r["group"]), str(r["dataset"])): (str(r["A_flag"]), str(r["B_flag"]))
    for _, r in et_flags.iterrows()
}

def get_flag(row, lookup):
    k = (str(row["scenario"])[:60], str(row["group"]), str(row["dataset_norm"]))
    return lookup.get(k, ("ok", "ok"))

all_df[["se_A_flag", "se_B_flag"]] = all_df.apply(
    lambda r: pd.Series(get_flag(r, se_lookup)), axis=1)
all_df[["et_A_flag", "et_B_flag"]] = all_df.apply(
    lambda r: pd.Series(get_flag(r, et_lookup)), axis=1)

# ── Keep only flagged CQs (at least one ABOVE or BELOW) ──────────────────────
is_flagged = (
    (all_df["se_A_flag"] != "ok") | (all_df["se_B_flag"] != "ok") |
    (all_df["et_A_flag"] != "ok") | (all_df["et_B_flag"] != "ok")
)
flagged_df = all_df[is_flagged].copy().reset_index(drop=True)
print(f"Flagged CQs in pool: {len(flagged_df)}")

# Summarise flag direction
def flag_dir(row):
    flags = [row["se_A_flag"], row["se_B_flag"], row["et_A_flag"], row["et_B_flag"]]
    if "ABOVE" in flags and "BELOW" in flags: return "MIXED"
    if "ABOVE" in flags: return "ABOVE"
    if "BELOW" in flags: return "BELOW"
    return "ok"

flagged_df["flag_direction"] = flagged_df.apply(flag_dir, axis=1)
print(flagged_df["flag_direction"].value_counts().to_string())
print()

# ── Stratified sampling: 4 non-overlapping subsets ───────────────────────────
# ~15 CQs per subset, balanced ABOVE/BELOW, spread across datasets and groups
# Strategy: sample within each (dataset, flag_direction) stratum, then round-robin

SEED = 77

strata = []
for (ds, fdir), grp in flagged_df.groupby(["dataset", "flag_direction"]):
    shuffled = grp.sample(frac=1, random_state=SEED).reset_index(drop=True)
    strata.append(shuffled)

# Interleave rows across strata to maximise diversity
interleaved = []
max_len = max(len(s) for s in strata)
for i in range(max_len):
    for s in strata:
        if i < len(s):
            interleaved.append(s.iloc[i])

pool = pd.DataFrame(interleaved).drop_duplicates(subset=["cq"]).reset_index(drop=True)

N_SUBSETS    = 4
TARGET_PER_S = 15

combined = pool.copy()
combined["_subset"] = [i % N_SUBSETS + 1 for i in range(len(combined))]

# ── Save per subset ──────────────────────────────────────────────────────────
ANNOTATOR_COLS = [
    "dataset",
    "group",
    "scenario",          # full scenario text
    "cq",
    # flag columns — shown to annotators so they know this is a flagged case
    "se_A_flag",         # SE prevalence flag: ABOVE / BELOW / ok
    "se_B_flag",         # SE intensity flag:  ABOVE / BELOW / ok
    "et_A_flag",         # eval-term prevalence flag
    "et_B_flag",         # eval-term intensity flag
    # blank columns to fill in
    "source_deviation",  # Y / N
    "framing_bias",      # Y / N
    "notes",
]

GOLD_COLS = ANNOTATOR_COLS + [
    "gold_source_deviation",
    "gold_framing_bias",
]

for s in range(1, N_SUBSETS + 1):
    sub_df = combined[combined["_subset"] == s].head(TARGET_PER_S).copy()
    sub_df = sub_df.sample(frac=1, random_state=SEED + s).reset_index(drop=True)
    for col in ["source_deviation", "framing_bias", "notes"]:
        sub_df[col] = ""

    ann_path  = RESULTS_DIR / f"flag_subset_{s}_annotator.csv"
    gold_path = RESULTS_DIR / f"flag_subset_{s}_with_gold.csv"
    sub_df[ANNOTATOR_COLS].to_csv(ann_path,  index=False)
    sub_df[GOLD_COLS].to_csv(gold_path, index=False)

    print(f"── Flag Subset {s} ({len(sub_df)} CQs) ──")
    print(f"  annotator → {ann_path}")
    print(f"  with gold → {gold_path}")
    print(f"  flag directions: {sub_df['flag_direction'].value_counts().to_dict()}")
    print(sub_df.groupby(["dataset", "group"]).size().rename("n").to_string())
    has_gold = (sub_df["gold_source_deviation"] != "").sum()
    print(f"  rows with gold labels: {has_gold}/{len(sub_df)}")
    print()

# ── Overlap check ─────────────────────────────────────────────────────────────
all_seen = []
for s in range(1, N_SUBSETS + 1):
    df_s = pd.read_csv(RESULTS_DIR / f"flag_subset_{s}_annotator.csv")
    df_s["_s"] = s
    all_seen.append(df_s)
all_seen_df = pd.concat(all_seen)
dupes = all_seen_df.groupby("cq")["_s"].nunique()
overlapping = dupes[dupes > 1]
if overlapping.empty:
    print("✓ No overlapping CQs across flag subsets.")
else:
    print(f"⚠ {len(overlapping)} CQs appear in multiple subsets!")
