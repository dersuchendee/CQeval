"""
Eval-Terms Deviation Analysis
------------------------------
Measures per-scenario prevalence (A) and intensity (B) of evaluative
terms in CQs for each generation group, reports median + IQR as the
normal range, and flags out-of-range scenarios.

For Polifonia scenarios only, also computes PROPAGATION:
  propagated  — scenario has eval terms AND CQ has eval terms
  introduced  — scenario has NO eval terms BUT CQ has eval terms
  avoided     — scenario has eval terms BUT CQ has NO eval terms
  clean       — scenario has NO eval terms AND CQ has NO eval terms

Groups
------
  Gold           human-written CQs
  OntoChat       GPT-4o single-step, closed LLM
  NeoNGPT_GPT4o  GPT-4o two-step NeOn, closed LLM
  NeoNGPT_Qwen3  Ollama Qwen3-14b two-step NeOn, open LLM

Outputs (results/)
------------------
  eval_terms_deviation_per_scenario.csv
  eval_terms_deviation_summary.csv
  eval_terms_deviation_flagged.csv
  eval_terms_propagation.csv          (Polifonia only)
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
    str(BASE_DIR / "data" / "annotated_neongpt_polifonia_eval_terms.csv")
)
NEONGPT_IB_WHOW_CVN  = pathlib.Path(
    str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_eval_terms.csv")
)
ONTOCHAT_IB_WHOW_CVN = pathlib.Path(
    str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_eval_terms.csv")
)
SCENARIO_ANNOTATIONS  = pathlib.Path(
    str(BASE_DIR / "data" / "")
    "polifonia_ontochat_cqs.csv"
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
    return not bool(_MULTI_CQ_BLOCK.search(str(cq_text)))


def parse_eval_terms(val):
    """
    Returns (count: int | None, is_valid: bool).
    "0"           → (0, True)
    "N: ..."      → (N, True)
    "N:term"      → (N, True)   [missing space]
    loose terms   → (n_terms, True)
    api_error/NaN → (None, False)
    """
    if pd.isna(val):
        return None, False
    s = str(val).strip()
    if s in ("", "api_error", "?", "nan"):
        return None, False
    if s == "0":
        return 0, True
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1)), True
    # no leading number — count comma/slash-separated terms
    terms = [t.strip() for t in re.split(r"[,/]+", s) if t.strip()]
    return (len(terms), True) if terms else (None, False)


def scenario_has_eval(val) -> bool:
    """True if the scenario-level annotation contains evaluative terms."""
    if pd.isna(val):
        return False
    return str(val).strip() not in ("", "-", "nan", "0")


# ── load scenario-level annotations (Polifonia only) ──────────────────────────
sc_df = pd.read_csv(SCENARIO_ANNOTATIONS)
sc_df = sc_df[~sc_df["Scenario"].apply(_excluded)]
# key: first 200 chars of scenario text → has_eval bool
sc_lookup = {
    str(row["Scenario"]).strip()[:200]: scenario_has_eval(row.get("eval_terms"))
    for _, row in sc_df.iterrows()
}

def lookup_scenario_eval(scenario_text: str):
    """Return True/False/None (None = no annotation available)."""
    key = str(scenario_text).strip()[:200]
    return sc_lookup.get(key, None)


# ── load & unify all CQ sources ───────────────────────────────────────────────
records = []

# 1. OntoChat Polifonia  (gold + OntoChat)
df = pd.read_csv(ONTOCHAT_POLIFONIA)
df = df[~df["Scenario"].apply(_excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group    = "Gold"     if src == "gold"    else "OntoChat"
    llm_type = "human"   if src == "gold"    else "closed"
    cq_text  = str(row.get("Question", "")).strip()
    cnt, valid = parse_eval_terms(row.get("eval_terms"))
    sc_eval = lookup_scenario_eval(row["Scenario"])
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                cq_text,
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "Polifonia",
        "eval_count":        cnt,
        "is_valid":          valid and _is_real_cq(cq_text),
        "scenario_has_eval": sc_eval,
    })

# 2. NeoNGPT Polifonia  (GPT4o + Qwen3)
df = pd.read_csv(NEONGPT_POLIFONIA)
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group    = "NeoNGPT_GPT4o" if src == "GPT4o" else "NeoNGPT_Qwen3"
    llm_type = "closed"        if src == "GPT4o" else "open"
    cq_text  = str(row.get("Question", "")).strip()
    cnt, valid = parse_eval_terms(row.get("eval_terms"))
    sc_eval = lookup_scenario_eval(row["Scenario"])
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                cq_text,
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "Polifonia",
        "eval_count":        cnt,
        "is_valid":          valid and _is_real_cq(cq_text),
        "scenario_has_eval": sc_eval,
    })

# 3. OntoChat IB/WHOW/CVN  (gold taken here only)
df = pd.read_csv(ONTOCHAT_IB_WHOW_CVN)
for _, row in df.iterrows():
    sys = str(row["System"]).strip()
    group    = "Gold"    if sys == "Gold_Standard_CQs" else "OntoChat"
    llm_type = "human"   if sys == "Gold_Standard_CQs" else "closed"
    cq_text  = str(row.get("CQ", "")).strip()
    cnt, valid = parse_eval_terms(row.get("eval_terms"))
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                cq_text,
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "IB_WHOW_CVN",
        "eval_count":        cnt,
        "is_valid":          valid and _is_real_cq(cq_text),
        "scenario_has_eval": None,
    })

# 4. NeoNGPT IB/WHOW/CVN  (gold excluded — already loaded above)
df = pd.read_csv(NEONGPT_IB_WHOW_CVN)
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys = str(row["System"]).strip()
    group    = "NeoNGPT_GPT4o" if sys == "GPT4o_CQs" else "NeoNGPT_Qwen3"
    llm_type = "closed"        if sys == "GPT4o_CQs" else "open"
    cq_text  = str(row.get("CQ", "")).strip()
    cnt, valid = parse_eval_terms(row.get("eval_terms"))
    records.append({
        "scenario":          str(row["Scenario"]).strip(),
        "cq":                cq_text,
        "group":             group,
        "llm_type":          llm_type,
        "dataset":           "IB_WHOW_CVN",
        "eval_count":        cnt,
        "is_valid":          valid and _is_real_cq(cq_text),
        "scenario_has_eval": None,
    })

all_df = pd.DataFrame(records)
valid_df = all_df[all_df["is_valid"]].copy()

print(f"Total records: {len(all_df)}  |  Valid: {len(valid_df)}")
print(valid_df.groupby(["group", "llm_type", "dataset"])["eval_count"].count().rename("n_valid"))

# ── per-scenario measures ──────────────────────────────────────────────────────
scenario_stats = (
    valid_df
    .groupby(["scenario", "group", "llm_type", "dataset"], sort=False)
    .agg(
        n_valid        = ("eval_count", "count"),
        n_with_eval    = ("eval_count", lambda x: (x > 0).sum()),
        sum_count      = ("eval_count", "sum"),
    )
    .reset_index()
)
scenario_stats["A_prevalence"] = scenario_stats["n_with_eval"] / scenario_stats["n_valid"]
scenario_stats["B_intensity"]  = scenario_stats["sum_count"]   / scenario_stats["n_valid"]
scenario_stats["scenario_short"] = scenario_stats["scenario"].str[:60]

# ── summary: median + IQR per group ───────────────────────────────────────────
summary_rows = []
for (group, llm_type), sub in scenario_stats.groupby(["group", "llm_type"]):
    for label, col in [("A_prevalence", "A"), ("B_intensity", "B")]:
        s = sub[label].dropna()
        if not len(s):
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

# ── flag out-of-range scenarios ───────────────────────────────────────────────
refs = {}
for _, row in summary_df.iterrows():
    refs.setdefault(row["group"], {})[row["measure"]] = {
        "Q1": row["Q1"], "Q3": row["Q3"]
    }

def flag(val, q1, q3):
    if pd.isna(val): return ""
    if val < q1:     return "BELOW"
    if val > q3:     return "ABOVE"
    return "ok"

flag_rows = []
for _, r in scenario_stats.iterrows():
    g  = r["group"]
    ra = refs.get(g, {}).get("A", {})
    rb = refs.get(g, {}).get("B", {})
    flag_rows.append({
        "scenario":  r["scenario_short"],
        "dataset":   r["dataset"],
        "group":     g,
        "llm_type":  r["llm_type"],
        "n_valid":   r["n_valid"],
        "A":         round(r["A_prevalence"], 3),
        "A_flag":    flag(r["A_prevalence"], ra.get("Q1", np.nan), ra.get("Q3", np.nan)),
        "B":         round(r["B_intensity"],  3),
        "B_flag":    flag(r["B_intensity"],  rb.get("Q1", np.nan), rb.get("Q3", np.nan)),
    })
flagged_df = pd.DataFrame(flag_rows)

# ── propagation analysis (Polifonia only) ─────────────────────────────────────
pol_valid = valid_df[
    (valid_df["dataset"] == "Polifonia") &
    (valid_df["scenario_has_eval"].notna())
].copy()

pol_valid["cq_has_eval"]   = pol_valid["eval_count"] > 0
pol_valid["sc_has_eval"]   = pol_valid["scenario_has_eval"].astype(bool)

def propagation_class(row):
    if row["sc_has_eval"]  and row["cq_has_eval"]:  return "propagated"
    if not row["sc_has_eval"] and row["cq_has_eval"]: return "introduced"
    if row["sc_has_eval"]  and not row["cq_has_eval"]: return "avoided"
    return "clean"

pol_valid["prop_class"] = pol_valid.apply(propagation_class, axis=1)

prop_summary = (
    pol_valid
    .groupby(["group", "llm_type", "prop_class"])
    .size()
    .reset_index(name="n_cqs")
)
prop_totals = pol_valid.groupby(["group", "llm_type"]).size().reset_index(name="total")
prop_summary = prop_summary.merge(prop_totals, on=["group", "llm_type"])
prop_summary["proportion"] = (prop_summary["n_cqs"] / prop_summary["total"]).round(3)

# also per-scenario propagation rates
prop_scenario = (
    pol_valid
    .groupby(["scenario", "group", "llm_type"])
    .agg(
        n_propagated  = ("prop_class", lambda x: (x == "propagated").sum()),
        n_introduced  = ("prop_class", lambda x: (x == "introduced").sum()),
        n_avoided     = ("prop_class", lambda x: (x == "avoided").sum()),
        n_clean       = ("prop_class", lambda x: (x == "clean").sum()),
        n_total       = ("prop_class", "count"),
        sc_has_eval   = ("sc_has_eval", "first"),
    )
    .reset_index()
)
prop_scenario["intro_rate"] = (
    prop_scenario["n_introduced"] / prop_scenario["n_total"]
).round(3)
prop_scenario["prop_rate"] = (
    prop_scenario.apply(
        lambda r: r["n_propagated"] / r["n_total"] if r["sc_has_eval"] else np.nan,
        axis=1
    )
).round(3)
prop_scenario["scenario_short"] = prop_scenario["scenario"].str[:60]

# ── save ───────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(exist_ok=True)
scenario_stats.to_csv(RESULTS_DIR / "eval_terms_deviation_per_scenario.csv", index=False)
summary_df.to_csv(RESULTS_DIR / "eval_terms_deviation_summary.csv", index=False)
flagged_df.to_csv(RESULTS_DIR / "eval_terms_deviation_flagged.csv", index=False)
prop_summary.to_csv(RESULTS_DIR / "eval_terms_propagation_summary.csv", index=False)
prop_scenario.to_csv(RESULTS_DIR / "eval_terms_propagation_per_scenario.csv", index=False)

# ── print ──────────────────────────────────────────────────────────────────────
ORDER = ["Gold", "OntoChat", "NeoNGPT_GPT4o", "NeoNGPT_Qwen3"]
summary_df["_o"] = summary_df["group"].apply(lambda g: ORDER.index(g) if g in ORDER else 99)
summary_df = summary_df.sort_values(["_o", "measure"]).drop(columns="_o")

print("\n\n╔══════════════════════════════════════════════════════════════╗")
print("║   EVAL-TERMS DEVIATION — NORMAL RANGE (median + IQR)         ║")
print("╚══════════════════════════════════════════════════════════════╝\n")
for label, mkey in [
    ("A  — Prevalence  (proportion of CQs with ≥1 eval term)", "A"),
    ("B  — Intensity   (mean eval terms per CQ)",               "B"),
]:
    print(f"── {label} ──\n")
    sub = summary_df[summary_df["measure"] == mkey][[
        "group", "llm_type", "n_scenarios", "median", "Q1", "Q3", "IQR", "min", "max"
    ]]
    print(sub.to_string(index=False))
    print()

print("\n── Flags per group ──\n")
for grp, sub in flagged_df.groupby("group"):
    aa = (sub["A_flag"]=="ABOVE").sum(); ab = (sub["A_flag"]=="BELOW").sum()
    ba = (sub["B_flag"]=="ABOVE").sum(); bb = (sub["B_flag"]=="BELOW").sum()
    print(f"{grp:20s}  n={len(sub)}  A_ABOVE={aa}  A_BELOW={ab}  B_ABOVE={ba}  B_BELOW={bb}")

print("\n── Flagged scenarios ──\n")
flags = flagged_df[(flagged_df["A_flag"] != "ok") | (flagged_df["B_flag"] != "ok")]
flags = flags.sort_values("group")
print(flags[["group","scenario","dataset","A","A_flag","B","B_flag"]].to_string(index=False))

print(f"\nTotal flagged: {len(flags)} / {len(flagged_df)}")

print("\n\n╔══════════════════════════════════════════════════════════════╗")
print("║   PROPAGATION (Polifonia only)                                ║")
print("╚══════════════════════════════════════════════════════════════╝\n")
prop_summary["_o"] = prop_summary["group"].apply(lambda g: ORDER.index(g) if g in ORDER else 99)
prop_summary = prop_summary.sort_values(["_o", "prop_class"]).drop(columns="_o")
print(prop_summary[["group","llm_type","prop_class","n_cqs","total","proportion"]].to_string(index=False))

print("\n── Per-scenario introduction rate (scenario has NO eval terms, CQ does) ──\n")
intro = prop_scenario[prop_scenario["sc_has_eval"] == False][
    ["scenario_short","group","n_introduced","n_total","intro_rate"]
].sort_values(["group","intro_rate"], ascending=[True, False])
print(intro.to_string(index=False))

print("\n── Per-scenario propagation rate (scenario HAS eval terms) ──\n")
prop_pos = prop_scenario[prop_scenario["sc_has_eval"] == True][
    ["scenario_short","group","n_propagated","n_total","prop_rate"]
].sort_values(["group","prop_rate"], ascending=[True, False])
print(prop_pos.to_string(index=False))
