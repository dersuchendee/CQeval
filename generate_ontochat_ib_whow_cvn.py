"""
IB / WHOW / CVN CQ Generator – OntoChat methodology
----------------------------------------------------
Reads benchmarkdataset_ib_whow_cvn.csv, groups by unique Scenario,
collects the associated gold-standard CQs, then generates new CQs
using OntoChat (Gradio: b289zhan/OntoChat) – writes 'ontochat failed' on error.

Output: output_ontochat_ib_whow_cvn.csv

Usage:
  python generate_ontochat_ib_whow_cvn.py
"""

import os
import sys
import time
import logging
import pathlib
import httpx
import pandas as pd

try:
    from gradio_client import Client
except ImportError:
    Client = None

# ─────────────────────────── paths ───────────────────────────────
BENCHMARK_CSV = pathlib.Path(str(BASE_DIR / "data" / "benchmarkdataset_ib_whow_cvn.csv"))
OUTPUT_CSV    = pathlib.Path(str(BASE_DIR / "output_ontochat_ib_whow_cvn.csv"))

# ─────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────── prompt builder ──────────────────────
def _prompt_for(scenario: str) -> str:
    return (
        "Here is a user scenario for ontology engineering:\n"
        f"\"\"\"\n{scenario}\n\"\"\"\n\n"
        "Please generate up to five competency questions for this scenario "
        "and return them on one single line, separated by a semicolon (;). "
        "Output only the questions, nothing else."
    )

# ─────────────────────────── OntoChat ────────────────────────────
_ONTOCHAT_CLIENT = None
_ONTOCHAT_KEY_STORED = False
_ONTOCHAT_INIT_RETRIES = int(os.getenv("ONTOCHAT_INIT_RETRIES", "4"))
_ONTOCHAT_INIT_WAIT    = float(os.getenv("ONTOCHAT_INIT_WAIT",    "5.0"))
_ONTOCHAT_HTTP_TIMEOUT = float(os.getenv("ONTOCHAT_HTTP_TIMEOUT", "60.0"))

HISTORY = [[None, "I am OntoChat, your conversational ontology engineering assistant. Here is the second step of the system. Please give me your user story and tell me how many competency questions you want me to generate from the user story."]]


def _prime_openai_key(client) -> None:
    global _ONTOCHAT_KEY_STORED
    api_key = os.getenv("OPENAI_API_KEY")
    if _ONTOCHAT_KEY_STORED or not api_key:
        return
    try:
        client.predict(api_key, api_name="/set_openai_api_key")
        _ONTOCHAT_KEY_STORED = True
        logger.info("✓ OpenAI key stored in OntoChat session")
    except Exception as exc:
        logger.warning(f"Could not set key in OntoChat: {exc}")


def _get_ontochat_client():
    global _ONTOCHAT_CLIENT
    if _ONTOCHAT_CLIENT is not None:
        return _ONTOCHAT_CLIENT
    if Client is None:
        raise RuntimeError("gradio_client not installed – run: pip install gradio_client")

    timeout = httpx.Timeout(_ONTOCHAT_HTTP_TIMEOUT)
    for attempt in range(1, _ONTOCHAT_INIT_RETRIES + 1):
        try:
            client = Client("b289zhan/OntoChat", httpx_kwargs={"timeout": timeout})
            _ONTOCHAT_CLIENT = client
            _prime_openai_key(client)
            return client
        except httpx.ReadTimeout as exc:
            logger.warning(f"OntoChat init timeout {attempt}/{_ONTOCHAT_INIT_RETRIES}: {exc}")
        except Exception as exc:
            logger.warning(f"OntoChat init attempt {attempt}/{_ONTOCHAT_INIT_RETRIES} failed: {exc}")
        if attempt < _ONTOCHAT_INIT_RETRIES:
            time.sleep(_ONTOCHAT_INIT_WAIT * attempt)

    raise RuntimeError("Unable to initialize OntoChat client after retries")


def generate_ontochat(prompt: str, retry: int = 3, wait: float = 8.0) -> str:
    try:
        client = _get_ontochat_client()
        for attempt in range(1, retry + 1):
            try:
                ans = client.predict(prompt, HISTORY, api_name="/cq_generator")[0]
                return ans.strip()
            except Exception as e:
                logger.warning(f"OntoChat query attempt {attempt}/{retry} failed: {e}")
                if attempt == retry:
                    raise
                time.sleep(wait * attempt)
    except Exception as e:
        logger.error(f"OntoChat failed: {e}")
        return "ontochat failed"

# ─────────────────────────── main logic ──────────────────────────
def main():
    # ── 1. Load dataset ───────────────────────────────────────────
    logger.info(f"Loading dataset from {BENCHMARK_CSV}")
    df = pd.read_csv(BENCHMARK_CSV)
    logger.info(f"Total rows: {len(df)}")

    # ── 2. Group by unique Scenario, collect gold-standard CQs ───
    groups = list(df.groupby("Scenario", dropna=False))
    logger.info(f"Unique scenarios: {len(groups)}")

    records = []
    for idx, (scenario, group) in enumerate(groups, start=1):
        project = group["Project Name"].iloc[0] if "Project Name" in group.columns else ""
        logger.info(f"[{idx}/{len(groups)}] [{project}] Processing: {str(scenario)[:70]}…")

        gold_cqs = [
            str(cq).strip()
            for cq in group["Competency Question"].dropna()
            if str(cq).strip()
        ]
        gold_combined = " ; ".join(gold_cqs)
        prompt = _prompt_for(str(scenario))

        # OntoChat
        logger.info(f"  → OntoChat")
        ontochat_result = generate_ontochat(prompt)

        records.append({
            "Project Name":      project,
            "Scenario":          scenario,
            "Gold_Standard_CQs": gold_combined,
            "OntoChat_CQs":      ontochat_result,
        })

        time.sleep(1)

    # ── 3. Write output CSV ───────────────────────────────────────
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"✓ Output written to {OUTPUT_CSV}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
