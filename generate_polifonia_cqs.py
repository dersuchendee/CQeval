"""
Polifonia CQ Generator
----------------------
Reads benchmarkdataset.csv, filters to Polifonia project rows,
groups by unique user story (Scenario), collects the associated
gold-standard CQs, then generates new CQs from three sources:
  1. OntoChat  (Gradio: b289zhan/OntoChat) – writes 'ontochat failed' on error
  2. GPT-4o    (OpenAI, temperature=0)
  3. Ollama    (qwen3:14b, temperature=0, local http://localhost:11434)

Output: output_ontochat_polifonia_cqs.csv

Source code adapted from:
  ./cq_generator_ontochat_app.py

Usage:
  export OPENAI_API_KEY=sk-...
  python generate_polifonia_cqs.py
"""

import os
import sys
import time
import logging
import pathlib
import httpx
import requests
import pandas as pd

try:
    from gradio_client import Client
except ImportError:
    Client = None

try:
    import openai
except ImportError:
    openai = None

# ─────────────────────────── paths ───────────────────────────────
BENCHMARK_CSV = pathlib.Path(
    str(BASE_DIR / "data" / "benchmarkdataset_polifonia.csv")
)
OUTPUT_CSV = pathlib.Path(str(BASE_DIR / "output_ontochat_polifonia_cqs.csv"))

OLLAMA_MODEL = "qwen3:14b"
OLLAMA_URL   = "http://localhost:11434/api/chat"

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

# ─────────────────────────── GPT-4o ──────────────────────────────
SYSTEM_PROMPT = (
    "You generate up to five competency questions for ontology engineering. "
    "Return them on one single line separated by ';'. Output only the questions."
)


def generate_gpt4o(prompt: str) -> str:
    if openai is None:
        return "Error: openai package not installed"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set"
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT-4o error: {e}"

# ─────────────────────────── Ollama ──────────────────────────────
def generate_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        return f"Ollama error: {e}"

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
        prompt = _prompt_for(str(scenario))

        # OntoChat
        logger.info(f"  → OntoChat")
        ontochat_result = generate_ontochat(prompt)

        # GPT-4o
        logger.info(f"  → GPT-4o")
        gpt4o_result = generate_gpt4o(prompt)

        # Ollama (qwen3:14b)
        logger.info(f"  → Ollama ({OLLAMA_MODEL})")
        ollama_result = generate_ollama(prompt)

        records.append({
            "Scenario":          scenario,
            "Gold_Standard_CQs": gold_combined,
            "OntoChat_CQs":      ontochat_result,
            "GPT4o_CQs":         gpt4o_result,
            f"Ollama_{OLLAMA_MODEL.replace(':', '_')}_CQs": ollama_result,
        })

        time.sleep(1)

    # ── 4. Write output CSV ───────────────────────────────────────
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"✓ Output written to {OUTPUT_CSV}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
