"""
Polifonia CQ Generator – NeonGPT methodology
----------------------------------------------
Reads benchmarkdataset.csv, filters to Polifonia project rows,
groups by unique user story (Scenario), collects the associated
gold-standard CQs, then generates new CQs using the NeOn two-step
prompting approach (exact prompts from restapi-for-neongpt/neongpt_api.py):
  Step 1: Generate ontology requirements from the scenario
  Step 2: Generate CQs from those requirements

Two generation sources:
  1. GPT-4o  (OpenAI, temperature=0, two-step)
  2. Ollama  (qwen3:14b, temperature=0, two-step, local http://localhost:11434)

Output: output_neongpt_polifonia_cqs.csv

Source prompts from:
  ./neongpt_api.py

Usage:
  export OPENAI_API_KEY=sk-...
  python generate_neongpt_cqs.py
"""

import os
import sys
import time
import logging
import pathlib
import requests
import pandas as pd

try:
    import openai
except ImportError:
    openai = None

# ─────────────────────────── paths ───────────────────────────────
BENCHMARK_CSV = pathlib.Path(
    str(BASE_DIR / "data" / "benchmarkdataset_polifonia.csv")
)
OUTPUT_CSV = pathlib.Path(str(BASE_DIR / "output_neongpt_polifonia_cqs.csv"))

OLLAMA_MODEL = "qwen3:14b"
OLLAMA_URL   = "http://localhost:11434/api/chat"

# ─────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────── NeOn prompts (exact from neongpt_api.py) ───────────
REQUIREMENTS_SYSTEM = "You are an ontology domain expert generating requirements."
CQ_SYSTEM           = "You are an ontology domain expert generating competency questions."

def _requirements_prompt(scenario: str) -> str:
    return (
        f"Scenario:\n{scenario}\n\n"
        "You are an experienced knowledge engineer. "
        "Following the NeOn methodology, first specify Purpose, Scope, "
        "Target Group, implementation language, intended uses, Functional "
        "and Non-functional Requirements for the ontology."
    )

def _cq_prompt(requirements: str) -> str:
    return (
        f"Based on the generated Specifications of Ontology Requirements "
        f"{requirements!r}, write a list of Competency Questions that the "
        "core module of the ontology should be able to answer. Make it as "
        "complete as possible, but focus on the core elements of the ontology. "
        "List the CQs semicolon-separated (CQ1 ; CQ2 ; …)."
    )

# ─────────────────────────── GPT-4o (two-step) ───────────────────
def generate_gpt4o(scenario: str) -> tuple[str, str]:
    """Returns (requirements, cqs)."""
    if openai is None:
        return "Error: openai package not installed", ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set", ""
    try:
        client = openai.OpenAI(api_key=api_key)

        # Step 1 – requirements
        r1 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": REQUIREMENTS_SYSTEM},
                {"role": "user",   "content": _requirements_prompt(scenario)},
            ],
            max_tokens=1000,
            temperature=0,
        )
        requirements = r1.choices[0].message.content.strip()

        # Step 2 – CQs
        r2 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CQ_SYSTEM},
                {"role": "user",   "content": _cq_prompt(requirements)},
            ],
            max_tokens=1000,
            temperature=0,
        )
        cqs = r2.choices[0].message.content.strip()
        return requirements, cqs

    except Exception as e:
        return f"GPT-4o error: {e}", ""

# ─────────────────────────── Ollama (two-step) ───────────────────
def _ollama_call(system: str, user: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_ollama(scenario: str) -> tuple[str, str]:
    """Returns (requirements, cqs)."""
    try:
        # Step 1 – requirements
        requirements = _ollama_call(REQUIREMENTS_SYSTEM, _requirements_prompt(scenario))

        # Step 2 – CQs
        cqs = _ollama_call(CQ_SYSTEM, _cq_prompt(requirements))
        return requirements, cqs

    except Exception as e:
        return f"Ollama error: {e}", ""

# ─────────────────────────── main logic ──────────────────────────
def main():
    # ── 1. Load dataset ───────────────────────────────────────────
    logger.info(f"Loading benchmark dataset from {BENCHMARK_CSV}")
    df = pd.read_csv(BENCHMARK_CSV)
    logger.info(f"Total rows: {len(df)}")

    # ── 2. Filter to Polifonia ────────────────────────────────────
    polifonia_df = df[df["Project Name"].str.strip().str.lower() == "polifonia"].copy()
    logger.info(f"Polifonia rows: {len(polifonia_df)}")

    if polifonia_df.empty:
        logger.error("No Polifonia rows found – check the Project Name column values.")
        sys.exit(1)

    # ── 3. Group by unique Scenario, collect gold-standard CQs ───
    groups = list(polifonia_df.groupby("Scenario", dropna=False))
    logger.info(f"Unique Polifonia scenarios: {len(groups)}")

    records = []
    for idx, (scenario, group) in enumerate(groups, start=1):
        logger.info(f"[{idx}/{len(groups)}] Processing: {str(scenario)[:70]}…")

        gold_cqs = [
            str(cq).strip()
            for cq in group["Competency Question"].dropna()
            if str(cq).strip()
        ]
        gold_combined = " ; ".join(gold_cqs)

        # GPT-4o (two-step)
        logger.info(f"  → GPT-4o (step 1: requirements)")
        gpt4o_reqs, gpt4o_cqs = generate_gpt4o(str(scenario))
        logger.info(f"  → GPT-4o (step 2: CQs done)")

        # Ollama qwen3:14b (two-step)
        logger.info(f"  → Ollama {OLLAMA_MODEL} (step 1: requirements)")
        ollama_reqs, ollama_cqs = generate_ollama(str(scenario))
        logger.info(f"  → Ollama {OLLAMA_MODEL} (step 2: CQs done)")

        records.append({
            "Scenario":                              scenario,
            "Gold_Standard_CQs":                    gold_combined,
            "GPT4o_Requirements":                   gpt4o_reqs,
            "GPT4o_CQs":                            gpt4o_cqs,
            f"Ollama_{OLLAMA_MODEL.replace(':', '_')}_Requirements": ollama_reqs,
            f"Ollama_{OLLAMA_MODEL.replace(':', '_')}_CQs":          ollama_cqs,
        })

        time.sleep(1)

    # ── 4. Write output CSV ───────────────────────────────────────
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"✓ Output written to {OUTPUT_CSV}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
