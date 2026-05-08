"""
Polifonia Eval-Terms Annotator
--------------------------------
Reads disaggregate_ontochat_neongpt_polifonia_cqs - ASEvalNeongpt.csv,
skips gold rows and empty CQ placeholders, and uses GPT-4o-mini with
in-context learning to fill in missing eval_terms annotations.

Existing annotations are preserved as-is.

Output: annotated_polifonia_eval_terms.csv
Cache:  results/polifonia_eval_terms_cache.json

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python annotate_polifonia_eval_terms.py
"""

import os
import re
import json
import pathlib
import logging
import pandas as pd
import anthropic

# ─────────────────────────── config ──────────────────────────────
INPUT_CSV    = BASE_DIR / "data" / "polifonia_neongpt_cqs.csv"
EXAMPLES_CSV = BASE_DIR / "data" / "ontochat_goldstandard_cqs.csv"
OUTPUT_CSV   = pathlib.Path(str(BASE_DIR / "data" / "annotated_polifonia_eval_terms.csv"))
CACHE_PATH   = BASE_DIR / "results" / "polifonia_eval_terms_cache.json"

MODEL              = "claude-haiku-4-5-20251001"
NONZERO_OVERSAMPLE = 10

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

# ─────────────────────────── CQ cleaning ─────────────────────────
_CQ_PREFIX = re.compile(r"^\s*CQ\s*\d+\s*[:\-–]?\s*", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(CQ\s*\d+\s*[;\s]*)+$", re.IGNORECASE)

def _clean_cq(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = _CQ_PREFIX.sub("", raw.strip()).strip()
    if not cleaned or _PLACEHOLDER.match(cleaned) or len(cleaned) < 5:
        return None
    return cleaned

def _needs_annotation(val) -> bool:
    if pd.isna(val):
        return True
    s = str(val).strip()
    return s in ("", "?", "nan")

# ─────────────────────────── normalise gold annotation ───────────
_COUNT_PREFIX = re.compile(r"^\d+\s*:\s*")

def _normalise(val: str) -> str:
    val = val.strip()
    if val == "0":
        return "0"
    if _COUNT_PREFIX.match(val):
        return _COUNT_PREFIX.sub(lambda m: m.group(0).replace(":", ": ").replace(":  ", ": "), val)
    terms = [t.strip() for t in re.split(r"[,/]", val) if t.strip()]
    return f"{len(terms)}: {', '.join(terms)}"

# ─────────────────────────── in-context examples ─────────────────
def _build_icl_examples() -> str:
    df = pd.read_csv(EXAMPLES_CSV)
    df = df.dropna(subset=["Question", "Scenario", "eval_terms"])
    df["eval_terms"] = df["eval_terms"].astype(str).str.strip()
    zero    = df[df["eval_terms"] == "0"].reset_index(drop=True)
    nonzero = df[df["eval_terms"] != "0"].reset_index(drop=True)
    nonzero_rep = pd.concat([nonzero] * NONZERO_OVERSAMPLE, ignore_index=True)
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
def annotate(scenario: str, cq: str, client: anthropic.Anthropic, cache: dict) -> str:
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
        resp = client.messages.create(
            model=MODEL,
            max_tokens=80,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        result = resp.content[0].text.strip()
        result = re.sub(r"(?i)^eval[_\s]terms\s*:\s*", "", result).strip()
    except Exception as e:
        logger.warning(f"API error: {e}")
        result = "api_error"

    cache[key] = result
    return result

# ─────────────────────────── main ────────────────────────────────
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    cache  = _load_cache()
    logger.info(f"Cache loaded: {len(cache)} entries")

    df = pd.read_csv(INPUT_CSV)

    # build output: only non-gold rows with real CQs
    mask = df.apply(
        lambda row: row.get("Source", "") != "gold"
                    and _clean_cq(str(row.get("Question", ""))) is not None,
        axis=1,
    )
    out_df = df[mask].copy()
    out_df["Question"] = out_df["Question"].apply(lambda q: _clean_cq(str(q)))

    to_annotate = out_df.index[out_df["eval_terms"].apply(_needs_annotation)].tolist()
    logger.info(f"Rows to annotate: {len(to_annotate)} / {len(out_df)}")

    for n, i in enumerate(to_annotate, start=1):
        row      = out_df.loc[i]
        scenario = str(row["Scenario"]).strip()
        cq       = str(row["Question"]).strip()
        logger.info(f"  [{n}/{len(to_annotate)}] [{row.get('Source','')}] {cq[:60]}…")
        out_df.at[i, "eval_terms"] = annotate(scenario, cq, client, cache)
        if n % 10 == 0:
            _save_cache(cache)

    _save_cache(cache)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"✓ Written {len(out_df)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
