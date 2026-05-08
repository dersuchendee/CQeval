"""
Eval-Terms Annotator
---------------------
Reads output_ontochat_ib_whow_cvn.csv and output_neongpt_ib_whow_cvn.csv,
expands them to 1 row per (Scenario, CQ) pair, and uses GPT-4o with
in-context learning (from the gold-standard annotation file) to annotate
eval_terms for each CQ.

eval_terms: evaluative, normative, or subjective words/phrases in the CQ
  that express a judgment or value stance (e.g. "credibility", "illogical",
  "non-elite", "unreliable", "accuracy", "relevant").

Outputs:
  annotated_ontochat_ib_whow_cvn_eval_terms.csv
  annotated_neongpt_ib_whow_cvn_eval_terms.csv

Cache: results/eval_terms_cache.json  (keyed by scenario+cq text)

Usage:
  export OPENAI_API_KEY=sk-...
  python annotate_eval_terms.py
"""

import os
import re
import json
import pathlib
import logging
import pandas as pd
from openai import OpenAI

# ─────────────────────────── config ──────────────────────────────
ONTOCHAT_CSV   = pathlib.Path(str(BASE_DIR / "output_ontochat_ib_whow_cvn.csv"))
NEONGPT_CSV    = pathlib.Path(str(BASE_DIR / "output_neongpt_ib_whow_cvn.csv"))
EXAMPLES_CSV   = BASE_DIR / "data" / "ontochat_goldstandard_cqs.csv"
OUT_ONTOCHAT   = pathlib.Path(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_eval_terms.csv"))
OUT_NEONGPT    = pathlib.Path(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_eval_terms.csv"))
CACHE_PATH     = BASE_DIR / "results" / "eval_terms_cache.json"

MODEL              = "gpt-4o"
NONZERO_OVERSAMPLE = 10   # repeat non-zero examples N times to balance classes

# ─────────────────────────── logging ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── cache ───────────────────────────────
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _cache_key(scenario: str, cq: str) -> str:
    return f"{scenario.strip()[:200]}|||{cq.strip()}"

# ─────────────────────────── normalise annotation value ──────────
_COUNT_PREFIX = re.compile(r"^\d+\s*:\s*")

def _normalise(val: str) -> str:
    """Ensure non-zero values have the 'N: term' format."""
    val = val.strip()
    if val == "0":
        return "0"
    # already has count prefix like "1: foo"
    if _COUNT_PREFIX.match(val):
        # fix missing space after colon
        return _COUNT_PREFIX.sub(lambda m: m.group(0).replace(":", ": ").replace(":  ", ": "), val)
    # missing count prefix — count comma-separated items
    terms = [t.strip() for t in re.split(r"[,/]", val) if t.strip()]
    return f"{len(terms)}: {', '.join(terms)}"

# ─────────────────────────── in-context examples ─────────────────
def _build_icl_examples() -> str:
    df = pd.read_csv(EXAMPLES_CSV)
    df = df.dropna(subset=["Question", "Scenario", "eval_terms"])
    df["eval_terms"] = df["eval_terms"].astype(str).str.strip()

    zero    = df[df["eval_terms"] == "0"].reset_index(drop=True)
    nonzero = df[df["eval_terms"] != "0"].reset_index(drop=True)

    # oversample non-zero examples to compensate for heavy class imbalance
    nonzero_rep = pd.concat([nonzero] * NONZERO_OVERSAMPLE, ignore_index=True)

    # interleave: alternate non-zero / zero
    n = max(len(zero), len(nonzero_rep))
    rows = []
    for i in range(n):
        if i < len(nonzero_rep): rows.append(nonzero_rep.iloc[i])
        if i < len(zero):        rows.append(zero.iloc[i])
    examples = pd.DataFrame(rows).reset_index(drop=True)

    lines = []
    for _, row in examples.iterrows():
        ann = _normalise(str(row["eval_terms"]))
        lines.append(
            f"Scenario: {str(row['Scenario']).strip()}\n"
            f"CQ: {str(row['Question']).strip()}\n"
            f"eval_terms: {ann}"
        )
    return "\n\n".join(lines)

ICL_EXAMPLES = _build_icl_examples()

SYSTEM_PROMPT = f"""You are an expert ontology engineer annotating competency questions (CQs) for evaluative language bias.

--- EXAMPLES (each shows a Scenario, a CQ, and the correct annotation) ---

{ICL_EXAMPLES}

--- END EXAMPLES ---

TASK
Given a Scenario and a CQ, identify words or phrases IN THE CQ that carry evaluative, normative, or subjective meaning — i.e. terms that express a judgment, quality assessment, or value stance rather than a neutral factual description.

WHAT COUNTS AS AN EVAL TERM
- Words implying quality or reliability: credible, accurate, illogical, unreliable, trustworthy, relevant, significant
- Words implying social/normative status: non-elite, elite, formal, recognized, legitimate
- Words implying desirability or value: effective, successful, optimal, appropriate, suitable
- Comparative or superlative judgments: better, best, worst, more important
- Terms like "best practice", "quality", "performance" used normatively

WHAT DOES NOT COUNT
- Purely factual or descriptive terms (names, dates, quantities, physical properties)
- Domain terminology that is neutral (e.g. "organ", "musician", "ontology", "NLP")
- Terms that merely describe a category without implicit judgment

RULES
1. Scan each word or phrase in the CQ.
2. Flag it ONLY if it carries clear evaluative or normative meaning.
3. Do NOT flag neutral domain terms.
4. Output ONLY the annotation value, nothing else (no "eval_terms:" label).

OUTPUT FORMAT (nothing else):
  0                         → no evaluative terms in the CQ
  N: term1, term2           → N evaluative terms found"""

# ─────────────────────────── LLM call ────────────────────────────
def annotate(scenario: str, cq: str, client: OpenAI, cache: dict) -> str:
    key = _cache_key(scenario, cq)
    if key in cache:
        return cache[key]

    user_msg = (
        f"Scenario: {scenario.strip()}\n\n"
        f"CQ: {cq.strip()}\n\n"
        "Which words or phrases in the CQ above carry evaluative, normative, or subjective meaning? "
        "Annotate eval_terms:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=80,
            temperature=0,
        )
        result = resp.choices[0].message.content.strip()
        # strip any leading "eval_terms:" label the model might add
        result = re.sub(r"(?i)^eval[_\s]terms\s*:\s*", "", result).strip()
    except Exception as e:
        logger.warning(f"API error: {e}")
        result = "api_error"

    cache[key] = result
    return result

# ─────────────────────────── CQ parsing ──────────────────────────
_CQ_PREFIX = re.compile(r"^\s*CQ\s*\d+\s*[:\-–]?\s*", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(CQ\s*\d+\s*[;\s]*)+$", re.IGNORECASE)

def _split_cqs(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [p.strip() for p in re.split(r"\s*;\s*", raw) if p.strip()]
    result = []
    for p in parts:
        if _PLACEHOLDER.match(p):
            continue
        cleaned = _CQ_PREFIX.sub("", p).strip()
        if not cleaned or _PLACEHOLDER.match(cleaned):
            continue
        if "Project Name" in cleaned or "Gold_Standard" in cleaned:
            continue
        if len(cleaned) < 5:
            continue
        result.append(cleaned)
    return result

# ─────────────────────────── file processing ─────────────────────
def process_file(input_csv: pathlib.Path, cq_columns: list[str],
                 output_csv: pathlib.Path, client: OpenAI, cache: dict) -> None:
    df = pd.read_csv(input_csv)
    records = []

    for _, row in df.iterrows():
        scenario = str(row["Scenario"]).strip()
        project  = str(row.get("Project Name", "")).strip()

        for col in cq_columns:
            raw = row.get(col, "")
            cqs = _split_cqs(str(raw) if pd.notna(raw) else "")
            for cq in cqs:
                records.append({
                    "Project Name": project,
                    "Scenario":     scenario,
                    "CQ":           cq,
                    "System":       col,
                    "eval_terms":   None,
                })

    total = len(records)
    logger.info(f"{input_csv.name} → {total} (scenario, CQ) pairs to annotate")

    for i, rec in enumerate(records, start=1):
        if rec["eval_terms"] is None:
            logger.info(f"  [{i}/{total}] [{rec['System']}] {rec['CQ'][:60]}…")
            rec["eval_terms"] = annotate(rec["Scenario"], rec["CQ"], client, cache)
            if i % 10 == 0:
                _save_cache(cache)

    _save_cache(cache)
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"✓ Written {len(out_df)} rows → {output_csv}")

# ─────────────────────────── main ────────────────────────────────
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key, timeout=60.0)
    cache  = _load_cache()
    logger.info(f"Cache loaded: {len(cache)} entries")

    process_file(
        ONTOCHAT_CSV,
        cq_columns=["Gold_Standard_CQs", "OntoChat_CQs"],
        output_csv=OUT_ONTOCHAT,
        client=client,
        cache=cache,
    )

    process_file(
        NEONGPT_CSV,
        cq_columns=["Gold_Standard_CQs", "GPT4o_CQs", "Ollama_qwen3_14b_CQs"],
        output_csv=OUT_NEONGPT,
        client=client,
        cache=cache,
    )

if __name__ == "__main__":
    main()
