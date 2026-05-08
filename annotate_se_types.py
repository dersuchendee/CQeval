"""
Superfluous Element Type Annotator
------------------------------------
Classifies each unique superfluous element into one of four deviation types:
  Expansion | Specification | Analytical Enrichment | Implicit Assumption

Uses Claude Haiku with the approved prompt. Needs context (scenario snippet)
so it samples the most frequent row for each element from the source files.

Output: results/unique_superfluous_elements_typed.csv
Cache:  results/se_types_cache.json
"""

import os
import re
import json
import pathlib
import logging
import pandas as pd
import anthropic

UNIQUE_SE_CSV = BASE_DIR / "results" / "unique_superfluous_elements.csv"
OUTPUT_CSV    = BASE_DIR / "results" / "unique_superfluous_elements_typed.csv"
CACHE_PATH    = BASE_DIR / "results" / "se_types_cache.json"
RESULTS_DIR   = BASE_DIR / "results"

MODEL = "claude-haiku-4-5-20251001"

EXCLUDE_PREFIXES = [
    "Persona\n\nOrtenz",
    "Successfully plan the restoration of an organ?",
    "The dashboard presents an overview",
]
_MULTI_CQ_BLOCK = re.compile(r"CQ\s*\d+\s*:.*CQ\s*\d+\s*:", re.DOTALL)

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert ontology engineer. Given a scenario, a CQ, and a superfluous element found in that CQ, classify the element into exactly ONE of the following deviation types:

1. Expansion — A new concept, action, or property introduced with NO basis in the scenario.
   e.g. "renovated" in a scenario about organ development.

2. Specification — A generic concept from the scenario made more concrete or typed.
   e.g. "places" when the scenario mentions general spatial context.

3. Analytical Enrichment — An analytical or interpretive dimension not present in the scenario (typically a metric, measure, or evaluation concept).
   e.g. "level of confidence" when the scenario discusses melody classification.

4. Implicit Assumption — An entity, role, or relation assumed by the CQ but neither stated nor justifiable from the scenario.
   e.g. "performer" assumed as a distinct role when the scenario only mentions "musician".

Rules:
- Classify based on the SUPERFLUOUS ELEMENT only, not the CQ as a whole.
- Output ONLY the category name, nothing else.

OUTPUT FORMAT (exactly one of these):
Expansion
Specification
Analytical Enrichment
Implicit Assumption"""

VALID_LABELS = {"Expansion", "Specification", "Analytical Enrichment", "Implicit Assumption"}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _excluded(s: str) -> bool:
    return any(str(s).startswith(p) for p in EXCLUDE_PREFIXES)

def _is_real_cq(t: str) -> bool:
    return not bool(_MULTI_CQ_BLOCK.search(str(t)))

def parse_se_elements(val) -> list[str]:
    if pd.isna(val): return []
    s = str(val).strip()
    if s in ("0", "No", ""): return []
    s = re.sub(r"^\d+\s*:\s*", "", s)
    return [e.strip().lower() for e in re.split(r"[,;/]+", s) if e.strip()]


def build_element_context_map() -> dict[str, tuple[str, str]]:
    """Return {element: (scenario_snippet, cq)} from all source files."""
    ctx: dict[str, tuple[str, str]] = {}

    def _add(scenario, cq, se_raw):
        if _excluded(scenario) or not _is_real_cq(cq):
            return
        for elem in parse_se_elements(se_raw):
            if elem not in ctx:
                ctx[elem] = (str(scenario).strip()[:500], str(cq).strip())

    # OntoChat Polifonia
    df = pd.read_csv(str(BASE_DIR / "data" / "")
                     "polifonia_ontochat_cqs.csv")
    for _, r in df.iterrows():
        _add(r.get("Scenario",""), r.get("Question",""), r.get("superfluous_elements", 0))

    # NeoNGPT Polifonia
    df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_polifonia_superfluous.csv"))
    for _, r in df.iterrows():
        _add(r.get("Scenario",""), r.get("Question",""), r.get("superfluous_elements", 0))

    # OntoChat IB/WHOW/CVN
    df = pd.read_csv(str(BASE_DIR / "data" / "annotated_ontochat_ib_whow_cvn_superfluous.csv"))
    for _, r in df.iterrows():
        _add(r.get("Scenario",""), r.get("CQ",""), r.get("superfluous_elements", 0))

    # NeoNGPT IB/WHOW/CVN
    df = pd.read_csv(str(BASE_DIR / "data" / "annotated_neongpt_ib_whow_cvn_superfluous.csv"))
    for _, r in df.iterrows():
        _add(r.get("Scenario",""), r.get("CQ",""), r.get("superfluous_elements", 0))

    return ctx


def annotate(element: str, scenario: str, cq: str,
             client: anthropic.Anthropic, cache: dict) -> str:
    key = element.strip().lower()
    if key in cache:
        return cache[key]

    user_msg = (
        f"Scenario:\n{scenario}\n\n"
        f"CQ:\n{cq}\n\n"
        f"Superfluous element to classify:\n{element}\n\n"
        "Output only the category name."
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=15,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        # match to valid label
        for label in VALID_LABELS:
            if label.lower() in raw.lower():
                result = label
                break
        else:
            result = "Expansion"   # safe fallback
            logger.warning(f"Unrecognised label '{raw}' for '{element}' → fallback Expansion")
    except Exception as e:
        logger.warning(f"API error for '{element}': {e}")
        result = "api_error"

    cache[key] = result
    return result


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    cache  = _load_cache()
    logger.info(f"Cache loaded: {len(cache)} entries")

    se_df = pd.read_csv(UNIQUE_SE_CSV)
    logger.info(f"Unique elements to classify: {len(se_df)}")

    logger.info("Building element → context map …")
    ctx_map = build_element_context_map()
    logger.info(f"Context map: {len(ctx_map)} elements with context")

    to_do = se_df[~se_df["superfluous_element"].str.lower().isin(cache)].index.tolist()
    logger.info(f"Need annotation: {len(to_do)}")

    for n, i in enumerate(to_do, 1):
        elem = str(se_df.at[i, "superfluous_element"]).strip()
        key  = elem.lower()
        scenario, cq = ctx_map.get(key, ("(no context available)", "(no context available)"))
        label = annotate(elem, scenario, cq, client, cache)
        logger.info(f"  [{n}/{len(to_do)}] {label:<25} | {elem}")
        if n % 30 == 0:
            _save_cache(cache)

    _save_cache(cache)

    se_df["deviation_type"] = se_df["superfluous_element"].str.lower().map(cache)
    se_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"✓ Written {len(se_df)} rows → {OUTPUT_CSV}")

    print("\n── Type distribution ──")
    print(se_df["deviation_type"].value_counts().to_string())
    print("\n── By frequency-weighted share ──")
    for dtype, grp in se_df.groupby("deviation_type"):
        print(f"  {dtype:<28} n={len(grp):4d}  total_freq={grp['frequency'].sum():5d}")


if __name__ == "__main__":
    main()
