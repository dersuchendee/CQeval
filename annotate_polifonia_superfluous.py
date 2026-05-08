"""
Polifonia Superfluous-Elements Annotator
-----------------------------------------
Reads disaggregate_ontochat_neongpt_polifonia_cqs - ASEvalNeongpt.csv,
skips gold rows and empty CQ placeholders, and uses Claude Haiku with
in-context learning to fill in missing superfluous_elements annotations.

Output: annotated_polifonia_superfluous.csv  (only annotated rows)
Cache:  results/polifonia_superfluous_cache.json

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python annotate_polifonia_superfluous.py
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
OUTPUT_CSV   = pathlib.Path(str(BASE_DIR / "data" / "annotated_polifonia_superfluous.csv"))
CACHE_PATH   = BASE_DIR / "results" / "polifonia_superfluous_cache.json"

MODEL = "claude-haiku-4-5-20251001"

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
    return str(val).strip() in ("", "?", "nan")

# ─────────────────────────── in-context examples ─────────────────
def _build_icl_examples() -> str:
    df = pd.read_csv(EXAMPLES_CSV)
    df = df.dropna(subset=["Question", "Scenario", "superfluous_elements"])
    neg = df[df["source_consistency"] == "No"].reset_index(drop=True)
    pos = df[df["source_consistency"] == "Yes"].reset_index(drop=True)
    n = max(len(neg), len(pos))
    rows = []
    for i in range(n):
        if i < len(neg): rows.append(neg.iloc[i])
        if i < len(pos): rows.append(pos.iloc[i])
    examples = pd.DataFrame(rows).reset_index(drop=True)
    lines = []
    for _, row in examples.iterrows():
        lines.append(
            f"Scenario: {str(row['Scenario']).strip()}\n"
            f"CQ: {str(row['Question']).strip()}\n"
            f"superfluous_elements: {str(row['superfluous_elements']).strip()}"
        )
    return "\n\n".join(lines)

ICL_EXAMPLES = _build_icl_examples()

SYSTEM_PROMPT = f"""You are an expert ontology engineer annotating competency questions (CQs) for source-consistency bias.

--- EXAMPLES (each shows a Scenario, a CQ, and the correct annotation) ---

{ICL_EXAMPLES}

--- END EXAMPLES ---

TASK
Given a Scenario and a CQ, find elements that appear IN THE CQ but are NOT present or implied in the Scenario.

RULES
1. Take each key word or concept from the CQ.
2. Check: does it appear in the scenario — verbatim or clearly implied?
3. Only flag it if you are CERTAIN it is absent from the scenario.
4. Never flag elements because they are absent from the CQ — direction is CQ → Scenario only.
5. Output ONLY the annotation value, nothing else (no "superfluous_elements:" label).

OUTPUT FORMAT (nothing else):
  0                         → all CQ elements are grounded in the scenario
  N: element1, element2     → N elements in the CQ are NOT in the scenario"""

# ─────────────────────────── LLM call ────────────────────────────
def annotate(scenario: str, cq: str, client: anthropic.Anthropic, cache: dict) -> str:
    key = _cache_key(scenario, cq)
    if key in cache:
        return cache[key]

    user_msg = (
        f"Scenario: {scenario.strip()}\n\n"
        f"CQ: {cq.strip()}\n\n"
        "Which words or concepts in the CQ above are NOT present or implied in the Scenario? "
        "Annotate superfluous_elements:"
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
        result = re.sub(r"(?i)^superfluous[_\s]elements\s*:\s*", "", result).strip()
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

    mask = df.apply(
        lambda row: row.get("Source", "") != "gold"
                    and _clean_cq(str(row.get("Question", ""))) is not None,
        axis=1,
    )
    out_df = df[mask].copy()
    out_df["Question"] = out_df["Question"].apply(lambda q: _clean_cq(str(q)))

    to_annotate = out_df.index[out_df["superfluous_elements"].apply(_needs_annotation)].tolist()
    logger.info(f"Rows to annotate: {len(to_annotate)} / {len(out_df)}")

    for n, i in enumerate(to_annotate, start=1):
        row      = out_df.loc[i]
        scenario = str(row["Scenario"]).strip()
        cq       = str(row["Question"]).strip()
        logger.info(f"  [{n}/{len(to_annotate)}] [{row.get('Source','')}] {cq[:60]}…")
        out_df.at[i, "superfluous_elements"] = annotate(scenario, cq, client, cache)
        if n % 10 == 0:
            _save_cache(cache)

    _save_cache(cache)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"✓ Written {len(out_df)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
