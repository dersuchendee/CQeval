"""
Superfluous Element Annotator
------------------------------
Reads output_ontochat_ib_whow_cvn.csv and output_neongpt_ib_whow_cvn.csv,
expands them to 1 row per (Scenario, CQ) pair, and uses GPT-4o-mini with
in-context learning (from the gold-standard annotation file) to annotate
superfluous_elements for each CQ.

Outputs:
  annotated_ontochat_ib_whow_cvn.csv
  annotated_neongpt_ib_whow_cvn.csv

Cache: results/superfluous_cache.json  (keyed by scenario+cq text)

Usage:
  export OPENAI_API_KEY=sk-...
  python annotate_superfluous.py
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
OUT_ONTOCHAT   = pathlib.Path(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv"))
OUT_NEONGPT    = pathlib.Path(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv"))
CACHE_PATH     = BASE_DIR / "results" / "superfluous_cache.json"

MODEL          = "gpt-4o"
N_ICL_EXAMPLES = None  # use all available examples

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

# ─────────────────────────── in-context examples ─────────────────
def _build_icl_examples() -> str:
    df = pd.read_csv(EXAMPLES_CSV)
    df = df.dropna(subset=["Question", "Scenario", "superfluous_elements"])
    # interleave negative and positive so examples alternate
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
        superf = str(row["superfluous_elements"]).strip()
        lines.append(
            f"Scenario: {str(row['Scenario']).strip()}\n"
            f"CQ: {str(row['Question']).strip()}\n"
            f"superfluous_elements: {superf}"
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
def annotate(scenario: str, cq: str, client: OpenAI, cache: dict) -> str:
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
        # skip anything that looks like a column header dumped as a value
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
        scenario   = str(row["Scenario"]).strip()
        project    = str(row.get("Project Name", "")).strip()

        for col in cq_columns:
            raw = row.get(col, "")
            cqs = _split_cqs(str(raw) if pd.notna(raw) else "")
            for cq in cqs:
                records.append({
                    "Project Name":        project,
                    "Scenario":            scenario,
                    "CQ":                  cq,
                    "System":              col,
                    "superfluous_elements": None,
                })

    total = len(records)
    logger.info(f"{input_csv.name} → {total} (scenario, CQ) pairs to annotate")

    for i, rec in enumerate(records, start=1):
        if rec["superfluous_elements"] is None:
            logger.info(f"  [{i}/{total}] [{rec['System']}] {rec['CQ'][:60]}…")
            rec["superfluous_elements"] = annotate(
                rec["Scenario"], rec["CQ"], client, cache
            )
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

    # OntoChat file: Gold + OntoChat CQs
    process_file(
        ONTOCHAT_CSV,
        cq_columns=["Gold_Standard_CQs", "OntoChat_CQs"],
        output_csv=OUT_ONTOCHAT,
        client=client,
        cache=cache,
    )

    # NeoNGPT file: Gold + GPT4o + Ollama CQs
    process_file(
        NEONGPT_CSV,
        cq_columns=["Gold_Standard_CQs", "GPT4o_CQs", "Ollama_qwen3_14b_CQs"],
        output_csv=OUT_NEONGPT,
        client=client,
        cache=cache,
    )

if __name__ == "__main__":
    main()
