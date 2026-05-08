"""
SE Justification Annotator
---------------------------
For every CQ with superfluous elements (SE > 0), determines whether those
elements are justified (can be reasonably inferred from the scenario) or not.

Sources of truth, in priority order:
  1. source_consistency column (where filled by human annotator)
       "Yes"  (inconsistency present)  → se_justified = "No"
       "No"   (consistent)             → se_justified = "Yes"
  2. LLM annotation (Claude Haiku) for rows without human annotation

Output: results/se_justified_annotated.csv
Cache:  results/se_justified_cache.json

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python annotate_se_justified.py
"""

import os
import re
import json
import pathlib
import logging
import pandas as pd
import anthropic

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
OUTPUT_CSV   = BASE_DIR / "results" / "se_justified_annotated.csv"
CACHE_PATH   = BASE_DIR / "results" / "se_justified_cache.json"
RESULTS_DIR  = BASE_DIR / "results"

MODEL = "claude-haiku-4-5-20251001"

EXCLUDE_PREFIXES = [
    "Persona\n\nOrtenz",
    "Successfully plan the restoration of an organ?",
    "The dashboard presents an overview",
]

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s")
logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────────
def _excluded(scenario: str) -> bool:
    return any(str(scenario).startswith(p) for p in EXCLUDE_PREFIXES)

_MULTI_CQ_BLOCK = re.compile(r"CQ\s*\d+\s*:.*CQ\s*\d+\s*:", re.DOTALL)

def _is_real_cq(cq_text: str) -> bool:
    return not bool(_MULTI_CQ_BLOCK.search(str(cq_text)))

def parse_se_count(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in ("0", "No", ""):
        return 0
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1))
    terms = [t.strip() for t in re.split(r"[,/]+", s) if t.strip()]
    return len(terms)

def extract_se_elements(val) -> str:
    """Return the element list as a clean string, or ''."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s in ("0", "No", ""):
        return ""
    # strip leading count "N: "
    s = re.sub(r"^\d+\s*:\s*", "", s).strip()
    return s

# ── cache ──────────────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _cache_key(scenario: str, cq: str, se_elements: str) -> str:
    return f"{scenario.strip()[:200]}|||{cq.strip()[:200]}|||{se_elements.strip()}"

# ── LLM prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert ontology engineer evaluating whether concepts in a competency question (CQ) are justified by a scenario description.

TASK
Given a Scenario, a CQ, and a list of superfluous elements (concepts found in the CQ that are not verbatim in the scenario), decide whether each element can be REASONABLY INFERRED or IMPLIED from reading the scenario alone.

LABELS
  Justified     — the element is a natural/logical inference from the scenario
  Not Justified — the element has NO grounding in the scenario whatsoever

RULES
1. Read the full scenario carefully.
2. For each listed superfluous element, ask: "Could a careful reader reasonably infer this from the scenario?"
3. Give ONE overall verdict for ALL the listed elements together.
4. Output ONLY the label, nothing else.

OUTPUT FORMAT (nothing else):
  Justified
  Not Justified"""


def annotate_llm(scenario: str, cq: str, se_elements: str,
                 client: anthropic.Anthropic, cache: dict) -> str:
    key = _cache_key(scenario, cq, se_elements)
    if key in cache:
        return cache[key]

    user_msg = (
        f"Scenario:\n{scenario.strip()}\n\n"
        f"CQ:\n{cq.strip()}\n\n"
        f"Superfluous elements (not verbatim in scenario):\n{se_elements.strip()}\n\n"
        "Are these elements justified by (inferrable from) the scenario? "
        "Answer with exactly two words: Justified / Not Justified"
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=10,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        if re.search(r"\bjustified\b", raw, re.I) and not re.search(r"\bnot\b", raw, re.I):
            result = "Justified"
        else:
            result = "Not Justified"
    except Exception as e:
        logger.warning(f"API error: {e}")
        result = "api_error"

    cache[key] = result
    return result


# ── load all datasets ──────────────────────────────────────────────────────────
records = []

# 1. OntoChat Polifonia  (gold + OntoChat)
df = pd.read_csv(ONTOCHAT_POLIFONIA)
df = df[~df["Scenario"].apply(_excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group = "Gold" if src == "gold" else "OntoChat"
    scenario = str(row.get("Scenario", "")).strip()
    cq       = str(row.get("Question", "")).strip()
    se_raw   = row.get("superfluous_elements", 0)
    se_count = parse_se_count(se_raw)
    se_elems = extract_se_elements(se_raw)
    sc       = str(row.get("source_consistency", "")).strip() if pd.notna(row.get("source_consistency")) else ""
    if not _is_real_cq(cq):
        continue
    records.append({
        "dataset":           "Polifonia",
        "group":             group,
        "scenario":          scenario,
        "cq":                cq,
        "se_raw":            str(se_raw),
        "se_count":          se_count,
        "se_elements":       se_elems,
        "source_consistency": sc,
        "se_justified":      None,
        "se_justified_src":  None,
    })

# 2. NeoNGPT Polifonia
df = pd.read_csv(NEONGPT_POLIFONIA)
df = df[~df["Scenario"].apply(_excluded)]
for _, row in df.iterrows():
    src = str(row["Source"]).strip()
    group = "NeoNGPT_GPT4o" if src == "GPT4o" else "NeoNGPT_Qwen3"
    scenario = str(row.get("Scenario", "")).strip()
    cq       = str(row.get("Question", "")).strip()
    se_raw   = row.get("superfluous_elements", 0)
    se_count = parse_se_count(se_raw)
    se_elems = extract_se_elements(se_raw)
    sc       = str(row.get("source_consistency", "")).strip() if pd.notna(row.get("source_consistency")) else ""
    if not _is_real_cq(cq):
        continue
    records.append({
        "dataset":           "Polifonia",
        "group":             group,
        "scenario":          scenario,
        "cq":                cq,
        "se_raw":            str(se_raw),
        "se_count":          se_count,
        "se_elements":       se_elems,
        "source_consistency": sc,
        "se_justified":      None,
        "se_justified_src":  None,
    })

# 3. OntoChat IB/WHOW/CVN
df = pd.read_csv(ONTOCHAT_IB_WHOW_CVN)
for _, row in df.iterrows():
    sys = str(row.get("System", "")).strip()
    group = "Gold" if sys == "Gold_Standard_CQs" else "OntoChat"
    scenario = str(row.get("Scenario", "")).strip()
    cq       = str(row.get("CQ", "")).strip()
    se_raw   = row.get("superfluous_elements", 0)
    se_count = parse_se_count(se_raw)
    se_elems = extract_se_elements(se_raw)
    if not _is_real_cq(cq):
        continue
    records.append({
        "dataset":           "IB_WHOW_CVN",
        "group":             group,
        "scenario":          scenario,
        "cq":                cq,
        "se_raw":            str(se_raw),
        "se_count":          se_count,
        "se_elements":       se_elems,
        "source_consistency": "",
        "se_justified":      None,
        "se_justified_src":  None,
    })

# 4. NeoNGPT IB/WHOW/CVN  (exclude gold to avoid double-counting)
df = pd.read_csv(NEONGPT_IB_WHOW_CVN)
df = df[df["System"] != "Gold_Standard_CQs"]
for _, row in df.iterrows():
    sys = str(row.get("System", "")).strip()
    group = "NeoNGPT_GPT4o" if sys == "GPT4o_CQs" else "NeoNGPT_Qwen3"
    scenario = str(row.get("Scenario", "")).strip()
    cq       = str(row.get("CQ", "")).strip()
    se_raw   = row.get("superfluous_elements", 0)
    se_count = parse_se_count(se_raw)
    se_elems = extract_se_elements(se_raw)
    if not _is_real_cq(cq):
        continue
    records.append({
        "dataset":           "IB_WHOW_CVN",
        "group":             group,
        "scenario":          scenario,
        "cq":                cq,
        "se_raw":            str(se_raw),
        "se_count":          se_count,
        "se_elements":       se_elems,
        "source_consistency": "",
        "se_justified":      None,
        "se_justified_src":  None,
    })

all_df = pd.DataFrame(records)
logger.info(f"Total records: {len(all_df)}")

# Keep only SE>0 rows for annotation
se_df = all_df[all_df["se_count"] > 0].copy()
logger.info(f"SE>0 rows to classify: {len(se_df)}")


# ── assign from source_consistency where available ─────────────────────────────
def derive_from_sc(row):
    sc = str(row["source_consistency"]).strip()
    if sc == "Yes":
        return "Not Justified", "source_consistency"
    elif sc == "No":
        return "Justified", "source_consistency"
    return None, None

for i, row in se_df.iterrows():
    label, src = derive_from_sc(row)
    if label:
        se_df.at[i, "se_justified"] = label
        se_df.at[i, "se_justified_src"] = src

need_llm = se_df[se_df["se_justified"].isna()]
logger.info(f"  From source_consistency: {len(se_df) - len(need_llm)} rows")
logger.info(f"  Need LLM annotation:     {len(need_llm)} rows")


# ── LLM annotation ─────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    cache  = _load_cache()
    logger.info(f"Cache loaded: {len(cache)} entries")

    to_annotate = se_df[se_df["se_justified"].isna()].index.tolist()
    logger.info(f"Running LLM annotation for {len(to_annotate)} rows …")

    for n, i in enumerate(to_annotate, start=1):
        row      = se_df.loc[i]
        scenario = row["scenario"]
        cq       = row["cq"]
        se_elems = row["se_elements"]
        label    = annotate_llm(scenario, cq, se_elems, client, cache)
        se_df.at[i, "se_justified"]     = label
        se_df.at[i, "se_justified_src"] = "llm"
        logger.info(f"  [{n}/{len(to_annotate)}] [{row['group']}] {label:10s} | {cq[:60]}")
        if n % 20 == 0:
            _save_cache(cache)

    _save_cache(cache)

    # ── save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    cols = ["dataset", "group", "scenario", "cq", "se_raw", "se_count",
            "se_elements", "source_consistency", "se_justified", "se_justified_src"]
    se_df[cols].to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✓ Written {len(se_df)} rows → {OUTPUT_CSV}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   SE JUSTIFICATION SUMMARY                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    ORDER = ["Gold", "OntoChat", "NeoNGPT_GPT4o", "NeoNGPT_Qwen3"]
    for group in ORDER:
        sub = se_df[se_df["group"] == group]
        if sub.empty:
            continue
        total = len(sub)
        vc    = sub["se_justified"].value_counts(dropna=False)
        just  = vc.get("Justified", 0)
        notj  = vc.get("Not Justified", 0)
        err   = vc.get("api_error", 0)
        print(f"{group:<20}  SE>0 CQs: {total:4d}  |  "
              f"Justified: {just:4d} ({just/total:.0%})  "
              f"Not Justified: {notj:4d} ({notj/total:.0%})"
              + (f"  errors: {err}" if err else ""))

    print()
    cross = (
        se_df.groupby(["group", "se_justified"])
        .size()
        .unstack(fill_value=0)
        .reindex(ORDER, fill_value=0)
    )
    print(cross.to_string())


if __name__ == "__main__":
    main()
