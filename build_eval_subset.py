"""
Build Evaluation Subset
-------------------------
Samples up to 10 CQs per dataset (Polifonia, IB, WHOW, CVN) for human review.
Includes OntoChat, NeoNGPT_GPT4o, NeoNGPT_Qwen3 (where available), and Gold
(all auto-annotated). Stratified by scenario.

Output: results/evaluation_subset.csv
"""

import re
import pathlib
import pandas as pd

RESULTS_DIR = BASE_DIR / "results"
OUTPUT_CSV  = RESULTS_DIR / "evaluation_subset.csv"

EXCLUDE_PREFIXES = [
    "Persona\n\nOrtenz",
    "Successfully plan the restoration of an organ?",
    "The dashboard presents an overview",
]
_MULTI_CQ_BLOCK = re.compile(r"CQ\s*\d+\s*:.*CQ\s*\d+\s*:", re.DOTALL)

def excluded(s): return any(str(s).startswith(p) for p in EXCLUDE_PREFIXES)
def is_real_cq(t): return not bool(_MULTI_CQ_BLOCK.search(str(t)))

def parse_se_count(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    if s in ("0","No",""): return 0
    m = re.match(r"^(\d+)", s)
    if m: return int(m.group(1))
    return len([t.strip() for t in re.split(r"[,/]+", s) if t.strip()])

records = []

# ── 1. OntoChat Polifonia (gold + OntoChat, human-annotated) ──────────────────
df = pd.read_csv(str(BASE_DIR / "data" / "")
                 "polifonia_ontochat_cqs.csv")
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group = "Gold" if src == "gold" else "OntoChat"
    cq = str(row.get("Question","")).strip()
    if not is_real_cq(cq): continue
    se_raw = row.get("superfluous_elements", 0)
    records.append({
        "dataset": "Polifonia", "group": group,
        "scenario": str(row["Scenario"]).strip(),
        "cq": cq,
        "superfluous_elements": str(se_raw) if pd.notna(se_raw) else "0",
        "eval_terms": str(row.get("eval_terms","0")) if pd.notna(row.get("eval_terms")) else "0",
        "annotation_source": "human",
        "se_count": parse_se_count(se_raw),
    })

# ── 2. NeoNGPT Polifonia ──────────────────────────────────────────────────────
df    = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_polifonia_superfluous.csv"))
df_et = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_polifonia_eval_terms.csv"))
et_map = {str(r["Scenario"])[:80]+"|||"+str(r["Question"])[:80]: str(r.get("eval_terms","0"))
          for _, r in df_et.iterrows()}
df = df[~df["Scenario"].apply(excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group = "NeoNGPT_GPT4o" if src=="GPT4o" else "NeoNGPT_Qwen3"
    cq = str(row.get("Question","")).strip()
    if not is_real_cq(cq): continue
    se_raw = row.get("superfluous_elements", 0)
    key = str(row["Scenario"])[:80]+"|||"+cq[:80]
    records.append({
        "dataset": "Polifonia", "group": group,
        "scenario": str(row["Scenario"]).strip(),
        "cq": cq,
        "superfluous_elements": str(se_raw) if pd.notna(se_raw) else "0",
        "eval_terms": et_map.get(key, "?"),
        "annotation_source": "llm",
        "se_count": parse_se_count(se_raw),
    })

# ── 3. OntoChat IB/WHOW/CVN ───────────────────────────────────────────────────
df    = pd.read_csv(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv"))
df_et = pd.read_csv(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_eval_terms.csv"))
et_map2 = {str(r["Scenario"])[:80]+"|||"+str(r.get("CQ",""))[:80]: str(r.get("eval_terms","0"))
           for _, r in df_et.iterrows()}
for _, row in df.iterrows():
    sys = str(row.get("System","")).strip()
    group = "Gold" if sys == "Gold_Standard_CQs" else "OntoChat"
    cq = str(row.get("CQ","")).strip()
    if not is_real_cq(cq): continue
    se_raw = row.get("superfluous_elements", 0)
    proj = str(row.get("Project Name","")).strip()
    key = str(row["Scenario"])[:80]+"|||"+cq[:80]
    records.append({
        "dataset": proj, "group": group,
        "scenario": str(row["Scenario"]).strip(),
        "cq": cq,
        "superfluous_elements": str(se_raw) if pd.notna(se_raw) else "0",
        "eval_terms": et_map2.get(key, "?"),
        "annotation_source": "llm",
        "se_count": parse_se_count(se_raw),
    })

# ── 4. NeoNGPT IB/WHOW/CVN (no gold — avoid double-counting) ─────────────────
df    = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv"))
df_et = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_eval_terms.csv"))
et_map3 = {str(r["Scenario"])[:80]+"|||"+str(r.get("CQ",""))[:80]: str(r.get("eval_terms","0"))
           for _, r in df_et.iterrows()}
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys = str(row.get("System","")).strip()
    group = "NeoNGPT_GPT4o" if sys=="GPT4o_CQs" else "NeoNGPT_Qwen3"
    cq = str(row.get("CQ","")).strip()
    if not is_real_cq(cq): continue
    se_raw = row.get("superfluous_elements", 0)
    proj = str(row.get("Project Name","")).strip()
    key = str(row["Scenario"])[:80]+"|||"+cq[:80]
    records.append({
        "dataset": proj, "group": group,
        "scenario": str(row["Scenario"]).strip(),
        "cq": cq,
        "superfluous_elements": str(se_raw) if pd.notna(se_raw) else "0",
        "eval_terms": et_map3.get(key, "?"),
        "annotation_source": "llm",
        "se_count": parse_se_count(se_raw),
    })

all_df = pd.DataFrame(records)

# Merge se_justified
sj = pd.read_csv(RESULTS_DIR / "se_justified_annotated.csv")[["group","cq","se_justified"]]
all_df = all_df.merge(sj, on=["group","cq"], how="left")
all_df["se_justified"] = all_df.apply(
    lambda r: "N/A" if r["se_count"] == 0
              else (r["se_justified"] if pd.notna(r["se_justified"]) else "?"),
    axis=1)

# ── Sampling: up to 10 per dataset, stratified by scenario + group ────────────
DATASET_ORDER   = ["Polifonia", "IntelligentBathrooms", "WHOW", "CVN"]
GROUP_PRIORITY  = ["OntoChat", "NeoNGPT_GPT4o", "NeoNGPT_Qwen3", "Gold"]
N_PER_DATASET   = 10
SEED            = 42

sampled_rows = []

for dataset in DATASET_ORDER:
    sub = all_df[all_df["dataset"] == dataset].copy()
    if sub.empty:
        continue

    scenarios = sub["scenario"].unique()
    candidates = []

    for scen in scenarios:
        scen_rows = sub[sub["scenario"] == scen]
        for grp in GROUP_PRIORITY:
            grp_rows = scen_rows[scen_rows["group"] == grp]
            if not grp_rows.empty:
                candidates.append(grp_rows.sample(1, random_state=SEED))

    if not candidates:
        continue

    pool = pd.concat(candidates).drop_duplicates(subset=["cq"])
    pool = pool.sample(frac=1, random_state=SEED)
    sampled_rows.append(pool.head(N_PER_DATASET))

eval_df = pd.concat(sampled_rows, ignore_index=True)

# ── Final column layout ───────────────────────────────────────────────────────
eval_df["scenario_short"] = eval_df["scenario"].str[:120]

out_cols = [
    "dataset",
    "group",
    "scenario_short",
    "cq",
    "annotation_source",
    # LLM-generated annotations to verify
    "superfluous_elements",
    "eval_terms",
    "se_justified",
    # Manual evaluation columns
    "superfluous_elements_correct",   # Y / N / Partial — was SE annotation accurate?
    "eval_terms_correct",             # Y / N / Partial — was eval terms annotation accurate?
    "se_justified_correct",           # Y / N — was Justified/Not Justified label correct?
    "notes",                          # free text
]

for col in ["superfluous_elements_correct","eval_terms_correct","se_justified_correct","notes"]:
    eval_df[col] = ""

RESULTS_DIR.mkdir(exist_ok=True)
eval_df[out_cols].to_csv(OUTPUT_CSV, index=False)

print(f"✓ Evaluation subset: {len(eval_df)} CQs → {OUTPUT_CSV}")
print()
print(eval_df.groupby(["dataset","group"]).size().rename("n").to_string())
