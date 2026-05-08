#!/usr/bin/env python3
"""
bias_analysis.py
================
Analyze bias annotations in scenario/user-story and competency-question (CQ) files.

Usage
-----
    python bias_analysis.py

Edit the CONFIGURATION section at the top before running.
All outputs are written to the folder specified by OUTPUT_DIR.
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import json
import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")                 # non-interactive backend; no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)

# ===========================================================================
# CONFIGURATION  ← edit this block before running
# ===========================================================================

# ── Input files ──────────────────────────────────────────────────────────────
SCENARIO_FILE  = "goldstandard-annotatedbias-inscenarios.csv"
SCENARIO_SHEET = ""    # Excel only: sheet name; leave "" for the first sheet

CQ_FILE  = "goldstandard-annotatedbias-inontochat.csv"
CQ_SHEET = ""          # Excel only: sheet name; leave "" for the first sheet

# ── Output directory (created automatically if absent) ───────────────────────
OUTPUT_DIR = "results"

# ── Scenario file column names ────────────────────────────────────────────────
SCENARIO_TEXT_COL         = "Scenario"
SCENARIO_FRAMING_BIAS_COL = "framing_bias"
SCENARIO_EVAL_TERMS_COL   = "eval_terms"
SCENARIO_STRUCTURAL_COL   = "structural_assumption"
SCENARIO_NOTES_COL        = "NOTES"

# ── CQ file column names ──────────────────────────────────────────────────────
CQ_ID_COL            = "ID"           # optional scenario ID column
CQ_SCENARIO_TEXT_COL = "Scenario"
CQ_TEXT_COL          = "Question"
CQ_GROUP_COL         = "Source"       # contains gold label or system name
CQ_SYSTEM_COL        = "Source"       # system/model name (same column in this dataset)
CQ_FRAMING_BIAS_COL  = "framing_bias"
CQ_EVAL_TERMS_COL    = "eval_terms"
CQ_STRUCTURAL_COL    = "structural_assumption"
CQ_SUPERFLUOUS_COL   = "superfluous_elements"
CQ_NOTES_COL         = "NOTES"

# ── Group labels ──────────────────────────────────────────────────────────────
GOLD_LABEL = "gold"      # value in CQ_GROUP_COL that identifies gold-standard CQs

# ── Scenario key matching ─────────────────────────────────────────────────────
SCENARIO_KEY_LEN = 100   # characters used to build cross-file matching keys

# ── Scenarios to exclude (matched via normalised key prefix) ─────────────────
# Add any prefix of a scenario's normalised key to skip it from all analysis.
EXCLUDED_SCENARIO_PREFIXES: List[str] = [
    "persona jorge",   # Jorge González archaeology scenario
]

# ── Claude API (for Part 3 — scenario_motivated classification) ───────────────
# Priority: environment variable > value set here.
# Export your key with:   export ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"   # fast model, sufficient for classification

# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# PART 1 — FILE LOADING
# ===========================================================================

def load_file(filepath: str, sheet: str = "") -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame.
    Auto-detects format from the file extension.
    If the file is Excel and no sheet name is given, loads the first sheet.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    if ext == ".csv":
        # dtype=str + keep_default_na=False avoids pandas auto-casting "0" to NaN
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    elif ext in (".xlsx", ".xls", ".xlsm", ".ods"):
        kw = {"sheet_name": sheet} if sheet else {"sheet_name": 0}
        df = pd.read_excel(filepath, dtype=str, keep_default_na=False, **kw)
        log.info(f"Excel: loaded sheet '{sheet or 'first'}' from {filepath}")
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Expected .csv or .xlsx.")

    # Normalise the two null-looking string variants to real NaN
    df.replace({"nan": np.nan, "": np.nan}, inplace=True)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {filepath}")
    return df


def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    optional: List[str],
    label: str,
) -> List[str]:
    """
    Assert required columns are present; warn about missing optional ones.
    Returns the subset of optional columns that actually exist in df.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}")

    present_opt: List[str] = []
    for col in optional:
        if col in df.columns:
            present_opt.append(col)
        else:
            log.warning(f"[{label}] Optional column '{col}' not found — related metrics will be skipped.")

    return present_opt


# ===========================================================================
# PART 2 — NORMALIZATION HELPERS
# ===========================================================================

# Strings that represent "no annotation" / empty
_NULL_TOKENS: Set[str] = {"-", "0", "none", "n/a", "na", "nan", "null", ""}

# Leading count annotations: "2: ", "1 (", "3:", etc.
_COUNT_PREFIX = re.compile(r"^\d+\s*[:(]\s*")
# Trailing ")" left after stripping "N ("
_TRAIL_PAREN  = re.compile(r"\)\s*$")
# Item separators: comma  semicolon  pipe  slash  newline  plus-sign
_SEPARATOR    = re.compile(r"[,;|/\n\r+]+")
# Purely numeric token (leftover count digit after splitting)
_ONLY_DIGITS  = re.compile(r"^\d+$")


def strip_count_prefix(s: str) -> str:
    """
    Strip a leading 'N:' or 'N (' count annotation from a cell string.
    Examples:
        "2: term1, term2"  → "term1, term2"
        "1 (renovated)"    → "renovated"
        "4: a, b, c, d"   → "a, b, c, d"
    """
    s = _COUNT_PREFIX.sub("", s.strip())
    s = _TRAIL_PAREN.sub("", s).strip()
    return s


def normalize_bool(value) -> Optional[bool]:
    """
    Convert Yes/No/True/False/1/0/- to bool.
    Returns None for missing, empty, or unrecognised values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return True
    if s in {"no", "false", "0", "n", "-"}:
        return False
    if s in _NULL_TOKENS:
        return None
    log.debug(f"Unrecognised boolean value: {value!r} — treated as None")
    return None


def normalize_text(text) -> str:
    """
    Lowercase, trim, collapse internal whitespace, strip trailing sentence
    punctuation. Safe to call on non-string input.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"[.!?]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def make_scenario_key(text, n: int = SCENARIO_KEY_LEN) -> str:
    """
    Build a short normalised key from scenario text for cross-file matching.
    Strips trailing punctuation that may differ between the two files.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = normalize_text(str(text))
    t = re.sub(r"[.!?,]", "", t)        # strip mid-sentence punctuation that sometimes differs
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def split_annotation(cell) -> Tuple[List[str], Set[str]]:
    """
    Parse an annotation cell into:
        raw_list  — original-case items (deduplicated, count prefixes stripped)
        norm_set  — lowercase-normalised items for exact set-matching

    Handles:
    • leading count prefixes ("2: item1, item2", "1 (term)")
    • separators:  ,  ;  |  /  newline  +
    • null placeholders:  -  0  nan  none  n/a
    • purely numeric leftover tokens
    • multi-word expressions (preserved as-is after splitting)
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return [], set()

    s = str(cell).strip()
    if normalize_text(s) in _NULL_TOKENS:
        return [], set()

    s = strip_count_prefix(s)
    if not s:
        return [], set()

    parts = _SEPARATOR.split(s)

    raw_list: List[str] = []
    seen_norm: Set[str] = set()
    norm_set:  Set[str] = set()

    for part in parts:
        part = part.strip("() \t")
        if not part:
            continue
        if _ONLY_DIGITS.match(part):
            continue                      # skip leftover count digits
        norm = normalize_text(part)
        if not norm or norm in _NULL_TOKENS:
            continue
        norm_set.add(norm)
        if norm not in seen_norm:         # deduplicate while preserving first-seen order
            seen_norm.add(norm)
            raw_list.append(part)

    return raw_list, norm_set


# ===========================================================================
# PART 3 — scenario_motivated CLASSIFICATION VIA CLAUDE API
# ===========================================================================

def _classify_single_note(notes_text: str, client) -> Tuple[str, str]:
    """
    Call Claude (temperature=0, structured output) to decide whether an
    annotator's NOTE indicates that a superfluous element in a CQ is
    motivated by the source scenario.

    Returns
    -------
    label  : "yes" | "no" | "uncertain"
    reason : one-sentence explanation from the model
    """
    notes_text = (notes_text or "").strip()
    if not notes_text:
        return "uncertain", "No annotator note available"

    prompt = (
        "You are classifying short annotator notes from a bias-analysis study on "
        "competency questions (CQs) derived from ontology user stories.\n\n"
        "A CQ has been annotated as containing a **superfluous element** — a term "
        "or concept not explicitly required by the source scenario.\n\n"
        f'The annotator\'s note is:\n"""\n{notes_text}\n"""\n\n'
        "Decide whether this note indicates that the superfluous element:\n"
        "- \"yes\"       : IS motivated or justifiable from the scenario context "
        "(e.g. 'justified by scenario', attributes element to scenario text, "
        "marks it 'Y', says it is derivable from the scenario).\n"
        "- \"no\"        : is NOT motivated by the scenario "
        "(e.g. 'not specified', 'not asked', 'introduced new word', "
        "'count not asked', or a plain question with no positive answer).\n"
        "- \"uncertain\" : the note is ambiguous, in a foreign language, "
        "an unanswered question, or otherwise insufficient to decide.\n\n"
        "Respond with ONLY valid JSON — no preamble, no markdown:\n"
        '{"label": "yes"|"no"|"uncertain", "reason": "<one short sentence>"}'
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=128,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        obj    = json.loads(raw)
        label  = str(obj.get("label", "uncertain")).lower().strip()
        if label not in {"yes", "no", "uncertain"}:
            label = "uncertain"
        reason = str(obj.get("reason", "")).strip()
        return label, reason
    except (json.JSONDecodeError, AttributeError):
        # Graceful fallback: look for the label string in the raw output
        low = raw.lower()
        if '"yes"' in low or "'yes'" in low:
            return "yes", raw[:120]
        if '"no"' in low or "'no'" in low:
            return "no", raw[:120]
        return "uncertain", raw[:120]


def add_scenario_motivated(
    df: pd.DataFrame,
    notes_col: str,
    superfluous_col: Optional[str],
) -> pd.DataFrame:
    """
    Add two columns to the CQ dataframe:
        scenario_motivated        : "yes" | "no" | "uncertain"
        scenario_motivated_reason : brief explanation string

    API calls are made only for rows with a non-zero superfluous-element
    annotation.  All other rows default to "uncertain" / "".
    """
    df = df.copy()
    df["scenario_motivated"]        = "uncertain"
    df["scenario_motivated_reason"] = ""

    if not superfluous_col or superfluous_col not in df.columns:
        log.warning("No superfluous_elements column found — scenario_motivated left as 'uncertain'.")
        return df

    def _has_superfluous(v) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        return normalize_text(str(v)) not in _NULL_TOKENS

    mask    = df[superfluous_col].apply(_has_superfluous)
    indices = df.index[mask].tolist()

    if not indices:
        log.info("No rows with superfluous elements — skipping API calls.")
        return df

    # Load classification cache (keyed by notes text) to avoid redundant API calls
    cache_path = Path(OUTPUT_DIR) / "scenario_motivated_cache.json"
    cache: Dict = {}
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            log.info(f"Loaded {len(cache)} cached scenario_motivated classifications.")
        except Exception:
            cache = {}

    # Identify which rows actually need a new API call
    to_classify = []
    for idx in indices:
        notes_val = df.at[idx, notes_col] if notes_col in df.columns else ""
        if notes_val is None or (isinstance(notes_val, float) and np.isnan(notes_val)):
            notes_val = ""
        notes_str = str(notes_val).strip()
        if notes_str in cache:
            df.at[idx, "scenario_motivated"]        = cache[notes_str]["label"]
            df.at[idx, "scenario_motivated_reason"] = cache[notes_str]["reason"]
        else:
            to_classify.append((idx, notes_str))

    if not to_classify:
        log.info("All scenario_motivated labels served from cache — no API calls needed.")
        return df

    log.info(f"Calling Claude API to classify scenario_motivated for {len(to_classify)} rows "
             f"({len(indices) - len(to_classify)} served from cache) …")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        return df
    except Exception as exc:
        log.error(f"Anthropic client init failed: {exc}")
        return df

    for i, (idx, notes_str) in enumerate(to_classify, 1):
        label, reason = _classify_single_note(notes_str, client)
        df.at[idx, "scenario_motivated"]        = label
        df.at[idx, "scenario_motivated_reason"] = reason
        cache[notes_str] = {"label": label, "reason": reason}
        if i % 10 == 0 or i == len(to_classify):
            log.info(f"  … classified {i}/{len(to_classify)} rows")

    # Persist updated cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning(f"Could not save classification cache: {exc}")

    log.info("scenario_motivated classification complete.")
    return df


# ===========================================================================
# ANNOTATION ENRICHMENT  (shared helper for both dataframes)
# ===========================================================================

def enrich_annotations(
    df: pd.DataFrame,
    fb_col: Optional[str],
    eval_col: Optional[str],
    struct_col: Optional[str],
    superfl_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse raw annotation columns and add boolean + split-annotation columns.

    Added columns
    -------------
    fb_bool       : bool | None
    eval_raw      : list[str]   — deduplicated raw items
    eval_norm     : set[str]    — normalised items for matching
    eval_count    : int
    struct_raw    : list[str]
    struct_norm   : set[str]
    struct_count  : int
    superfl_raw   : list[str]   (only if superfl_col given)
    superfl_norm  : set[str]
    superfl_count : int
    """
    df = df.copy()

    # framing_bias
    if fb_col and fb_col in df.columns:
        df["fb_bool"] = df[fb_col].apply(normalize_bool)
    else:
        df["fb_bool"] = None

    def _split_series(col):
        raw_col, norm_col = [], []
        for v in col:
            raw, norm = split_annotation(v)
            raw_col.append(raw)
            norm_col.append(norm)
        return raw_col, norm_col

    if eval_col and eval_col in df.columns:
        raw_list, norm_list = _split_series(df[eval_col])
        df["eval_raw"]   = raw_list
        df["eval_norm"]  = norm_list
        df["eval_count"] = [len(r) for r in raw_list]
    else:
        df["eval_raw"]   = [[] for _ in range(len(df))]
        df["eval_norm"]  = [set() for _ in range(len(df))]
        df["eval_count"] = 0

    if struct_col and struct_col in df.columns:
        raw_list, norm_list = _split_series(df[struct_col])
        df["struct_raw"]   = raw_list
        df["struct_norm"]  = norm_list
        df["struct_count"] = [len(r) for r in raw_list]
    else:
        df["struct_raw"]   = [[] for _ in range(len(df))]
        df["struct_norm"]  = [set() for _ in range(len(df))]
        df["struct_count"] = 0

    if superfl_col and superfl_col in df.columns:
        raw_list, norm_list = _split_series(df[superfl_col])
        df["superfl_raw"]   = raw_list
        df["superfl_norm"]  = norm_list
        df["superfl_count"] = [len(r) for r in raw_list]
    else:
        df["superfl_raw"]   = [[] for _ in range(len(df))]
        df["superfl_norm"]  = [set() for _ in range(len(df))]
        df["superfl_count"] = 0

    return df


def add_inconsistency_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append strict and weighted source-inconsistency columns.

    strict_source_inconsistency
        1 if superfl_count > 0 else 0

    weighted_source_inconsistency
        0.0  if superfl_count == 0
        0.5  if superfl_count > 0 AND scenario_motivated == "yes"
        1.0  if superfl_count > 0 AND scenario_motivated in {"no", "uncertain"}
    """
    df = df.copy()

    if "superfl_count" not in df.columns:
        df["superfl_count"] = 0

    df["strict_source_inconsistency"] = (df["superfl_count"] > 0).astype(int)

    def _weighted(row) -> float:
        if row["superfl_count"] == 0:
            return 0.0
        mot = str(row.get("scenario_motivated", "uncertain")).lower()
        return 0.5 if mot == "yes" else 1.0

    df["weighted_source_inconsistency"] = df.apply(_weighted, axis=1)
    return df


# ===========================================================================
# PART 4 — SCENARIO-LEVEL METRICS
# ===========================================================================

def compute_scenario_metrics(
    sc_df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Compute framing-related metrics for the scenario file.

    Returns
    -------
    metrics_dict   : dict of scalar metrics
    eval_freq_df   : frequency table of evaluative terms (scenarios)
    struct_freq_df : frequency table of structural assumptions (scenarios)
    """
    n = len(sc_df)
    if n == 0:
        return {}, pd.DataFrame(columns=["term", "frequency"]), pd.DataFrame(columns=["term", "frequency"])

    # framing_bias: None/NaN already filled with False upstream
    fb = sc_df["fb_bool"].fillna(False).astype(bool)

    n_fb        = int(fb.sum())
    n_eval      = int((sc_df["eval_count"] > 0).sum())
    n_struct    = int((sc_df["struct_count"] > 0).sum())
    tot_eval    = int(sc_df["eval_count"].sum())
    tot_struct  = int(sc_df["struct_count"].sum())

    biased_sc = sc_df[fb]

    metrics: Dict = {
        "total_scenarios":                   n,
        "n_framing_bias":                    n_fb,
        "prop_framing_bias":                 round(n_fb / n, 4),
        "n_with_eval_terms":                 n_eval,
        "prop_with_eval_terms":              round(n_eval / n, 4),
        "n_with_structural_assumption":      n_struct,
        "prop_with_structural_assumption":   round(n_struct / n, 4),
        "total_eval_count":                  tot_eval,
        "total_struct_count":                tot_struct,
        "mean_eval_per_scenario":            round(sc_df["eval_count"].mean(), 4),
        "mean_struct_per_scenario":          round(sc_df["struct_count"].mean(), 4),
        "mean_eval_framing_biased_only":     round(biased_sc["eval_count"].mean(), 4)
                                             if len(biased_sc) > 0 else None,
        "mean_struct_framing_biased_only":   round(biased_sc["struct_count"].mean(), 4)
                                             if len(biased_sc) > 0 else None,
    }

    # Frequency tables
    eval_ctr   = Counter()
    struct_ctr = Counter()
    for ns in sc_df["eval_norm"]:
        eval_ctr.update(ns)
    for ns in sc_df["struct_norm"]:
        struct_ctr.update(ns)

    eval_freq_df   = pd.DataFrame(eval_ctr.most_common(),   columns=["term", "frequency"])
    struct_freq_df = pd.DataFrame(struct_ctr.most_common(), columns=["term", "frequency"])

    return metrics, eval_freq_df, struct_freq_df


# ===========================================================================
# PART 5 — CQ-LEVEL METRICS
# ===========================================================================

def _group_metrics(grp: pd.DataFrame) -> Dict:
    """Compute CQ metrics for a single group slice."""
    n = len(grp)
    if n == 0:
        return {"total_cqs": 0}

    cqs_per_sc = grp.groupby("scenario_key").size()
    fb         = grp["fb_bool"].fillna(False).astype(bool)

    m: Dict = {
        "total_cqs":                   n,
        "distinct_scenarios":          int(grp["scenario_key"].nunique()),
        "mean_cqs_per_scenario":       round(float(cqs_per_sc.mean()),   4),
        "median_cqs_per_scenario":     round(float(cqs_per_sc.median()), 4),
        "max_cqs_per_scenario":        int(cqs_per_sc.max()),
        "n_framing_bias":              int(fb.sum()),
        "prop_framing_bias":           round(float(fb.mean()), 4),
        "n_with_eval_terms":           int((grp["eval_count"] > 0).sum()),
        "prop_with_eval_terms":        round(float((grp["eval_count"] > 0).mean()), 4),
        "n_with_struct_assumption":    int((grp["struct_count"] > 0).sum()),
        "prop_with_struct_assumption": round(float((grp["struct_count"] > 0).mean()), 4),
        "mean_eval_count":             round(float(grp["eval_count"].mean()), 4),
        "mean_struct_count":           round(float(grp["struct_count"].mean()), 4),
        "cooc_fb_eval":                int((fb & (grp["eval_count"] > 0)).sum()),
        "cooc_fb_struct":              int((fb & (grp["struct_count"] > 0)).sum()),
        "cooc_eval_struct":            int(((grp["eval_count"] > 0) & (grp["struct_count"] > 0)).sum()),
    }

    if "superfl_count" in grp.columns:
        n_sp = int((grp["superfl_count"] > 0).sum())
        m.update({
            "n_with_superfl":              n_sp,
            "prop_with_superfl":           round(n_sp / n, 4),
            "total_superfl_count":         int(grp["superfl_count"].sum()),
            "mean_superfl_count":          round(float(grp["superfl_count"].mean()), 4),
            "strict_inconsistency_rate":   round(float(grp["strict_source_inconsistency"].mean()), 4),
            "weighted_inconsistency_mean": round(float(grp["weighted_source_inconsistency"].mean()), 4),
        })

    return m


def compute_cq_metrics(
    cq_df: pd.DataFrame,
    systems: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Compute CQ metrics for: all / gold / generated / each system.

    Returns
    -------
    metrics_df  : DataFrame indexed by group name
    freq_tables : nested dict  { "eval"|"struct"|"superfl" : { group: freq_df } }
    """
    groups: Dict[str, pd.DataFrame] = {
        "all":       cq_df,
        "gold":      cq_df[cq_df["group"] == "gold"],
        "generated": cq_df[cq_df["group"] == "generated"],
    }
    for sys in systems:
        groups[sys] = cq_df[cq_df["system"] == sys]

    rows = []
    for gname, gdf in groups.items():
        m = _group_metrics(gdf)
        m["group"] = gname
        rows.append(m)

    metrics_df = pd.DataFrame(rows).set_index("group")

    # Frequency tables
    freq_tables: Dict[str, Dict[str, pd.DataFrame]] = {"eval": {}, "struct": {}, "superfl": {}}
    for col_key, df_col in [("eval", "eval_norm"), ("struct", "struct_norm"), ("superfl", "superfl_norm")]:
        for gname, gdf in groups.items():
            ctr = Counter()
            for ns in gdf[df_col]:
                ctr.update(ns)
            freq_tables[col_key][gname] = pd.DataFrame(
                ctr.most_common(), columns=["term", "frequency"]
            )

    return metrics_df, freq_tables


# ===========================================================================
# PART 6 — PER-SCENARIO CQ COUNTS: MICRO vs MACRO
# ===========================================================================

def compute_per_scenario_counts(
    cq_df: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Compute per-scenario statistics separately for generated and gold CQs,
    then aggregate both micro (CQ-weighted) and macro (scenario-weighted) metrics.

    Micro: every CQ counts equally (standard group mean over the full group).
    Macro: every scenario counts equally (average of per-scenario rates).

    Returns
    -------
    per_sc_results : { "generated": DataFrame, "gold": DataFrame }
    aggregates     : { "macro": {...}, "exposure": {...} }
    """
    per_sc_results: Dict[str, pd.DataFrame] = {}

    for grp_label in ("generated", "gold"):
        grp_df = cq_df[cq_df["group"] == grp_label]
        rows = []

        for sc_key, sc_grp in grp_df.groupby("scenario_key"):
            n_cq   = len(sc_grp)
            fb     = sc_grp["fb_bool"].fillna(False).astype(bool)
            n_fb   = int(fb.sum())

            row: Dict = {
                "scenario_key":              sc_key,
                "cq_count":                  n_cq,
                "fb_count":                  n_fb,
                "fb_rate":                   n_fb / n_cq,
                "eval_total":                int(sc_grp["eval_count"].sum()),
                "struct_total":              int(sc_grp["struct_count"].sum()),
                "superfl_total":             int(sc_grp["superfl_count"].sum()),
                "strict_inconsistency_rate": float(sc_grp["strict_source_inconsistency"].mean()),
                "weighted_inconsistency_mean": float(sc_grp["weighted_source_inconsistency"].mean()),
            }

            if "propagated_framing" in sc_grp.columns:
                valid_prop = sc_grp["propagated_framing"].dropna()
                row["propagated_count"] = int(valid_prop.sum())
                row["propagated_rate"]  = float(valid_prop.mean()) if len(valid_prop) > 0 else np.nan
            else:
                row["propagated_count"] = 0
                row["propagated_rate"]  = np.nan

            rows.append(row)

        per_sc_results[grp_label] = pd.DataFrame(rows)

    # Macro aggregates
    macro: Dict = {}
    for lbl in ("generated", "gold"):
        df = per_sc_results.get(lbl, pd.DataFrame())
        if df.empty:
            macro[lbl] = {}
            continue
        macro[lbl] = {
            "macro_mean_fb_rate":                    round(float(df["fb_rate"].mean()), 4),
            "macro_mean_strict_inconsistency_rate":  round(float(df["strict_inconsistency_rate"].mean()), 4),
            "macro_mean_weighted_inconsistency":     round(float(df["weighted_inconsistency_mean"].mean()), 4),
            "macro_mean_propagated_rate":            round(float(df["propagated_rate"].dropna().mean()), 4)
                                                     if "propagated_rate" in df.columns else None,
        }

    # Generated CQ exposure distribution
    gen_df  = per_sc_results.get("generated", pd.DataFrame())
    exposure: Dict = {}
    if not gen_df.empty:
        cnt_dist = gen_df["cq_count"].value_counts().sort_index()
        exposure = {
            "scenarios_with_1_cq":       int(cnt_dist.get(1, 0)),
            "scenarios_with_2_cqs":      int(cnt_dist.get(2, 0)),
            "scenarios_with_3plus_cqs":  int((gen_df["cq_count"] >= 3).sum()),
            "cq_count_distribution":     cnt_dist.to_dict(),
        }

    return per_sc_results, {"macro": macro, "exposure": exposure}


# ===========================================================================
# PART 7 — FRAMING-BIAS PROPAGATION FROM SCENARIO TO CQ
# ===========================================================================

def compute_propagation(
    cq_df: pd.DataFrame,
    sc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each CQ, compute framing-bias propagation from its linked scenario.

    Matching rule: exact intersection of normalised annotation item sets.
    Multi-word expressions are preserved; no fuzzy matching or embeddings.

    New columns added to cq_df
    --------------------------
    sc_eval_norm      : set[str]  — scenario evaluative-term set
    sc_struct_norm    : set[str]  — scenario structural-assumption set
    shared_eval       : set[str]  — intersection of CQ and scenario eval sets
    shared_struct     : set[str]  — intersection of CQ and scenario struct sets
    propagated_eval   : 0 | 1    — 1 if shared_eval is non-empty
    propagated_struct : 0 | 1    — 1 if shared_struct is non-empty
    propagated_framing: 0 | 1    — 1 if either propagated_eval or _struct is 1
    eval_precision    : float    — |shared_eval| / |cq_eval|  (NaN if cq_eval empty)
    eval_recall       : float    — |shared_eval| / |sc_eval|  (NaN if sc_eval empty)
    struct_precision  : float    — |shared_struct| / |cq_struct|
    struct_recall     : float    — |shared_struct| / |sc_struct|
    """
    # Build lookup: scenario_key → scenario row
    sc_lookup: Dict[str, pd.Series] = {}
    for _, row in sc_df.iterrows():
        k = row["scenario_key"]
        if k:
            sc_lookup[k] = row

    cq_df = cq_df.copy()

    sc_eval_list    = []
    sc_struct_list  = []
    shared_eval_l   = []
    shared_struct_l = []
    prop_eval_l     = []
    prop_struct_l   = []
    prop_framing_l  = []
    eval_prec_l     = []
    eval_rec_l      = []
    struct_prec_l   = []
    struct_rec_l    = []

    for _, row in cq_df.iterrows():
        sc = sc_lookup.get(row["scenario_key"])

        if sc is None:
            # No matching scenario — propagation metrics are NaN
            sc_eval_list.append(set())
            sc_struct_list.append(set())
            shared_eval_l.append(set())
            shared_struct_l.append(set())
            for lst in (prop_eval_l, prop_struct_l, prop_framing_l,
                        eval_prec_l, eval_rec_l, struct_prec_l, struct_rec_l):
                lst.append(np.nan)
            continue

        sc_eval   = sc["eval_norm"]
        sc_struct = sc["struct_norm"]
        cq_eval   = row["eval_norm"]
        cq_struct = row["struct_norm"]

        shared_eval   = sc_eval   & cq_eval
        shared_struct = sc_struct & cq_struct

        p_eval   = 1 if shared_eval   else 0
        p_struct = 1 if shared_struct else 0
        p_frame  = 1 if (p_eval or p_struct) else 0

        sc_eval_list.append(sc_eval)
        sc_struct_list.append(sc_struct)
        shared_eval_l.append(shared_eval)
        shared_struct_l.append(shared_struct)
        prop_eval_l.append(p_eval)
        prop_struct_l.append(p_struct)
        prop_framing_l.append(p_frame)

        eval_prec_l.append(len(shared_eval)   / len(cq_eval)   if cq_eval   else np.nan)
        eval_rec_l.append( len(shared_eval)   / len(sc_eval)   if sc_eval   else np.nan)
        struct_prec_l.append(len(shared_struct) / len(cq_struct) if cq_struct else np.nan)
        struct_rec_l.append( len(shared_struct) / len(sc_struct) if sc_struct  else np.nan)

    cq_df["sc_eval_norm"]      = sc_eval_list
    cq_df["sc_struct_norm"]    = sc_struct_list
    cq_df["shared_eval"]       = shared_eval_l
    cq_df["shared_struct"]     = shared_struct_l
    cq_df["propagated_eval"]   = prop_eval_l
    cq_df["propagated_struct"] = prop_struct_l
    cq_df["propagated_framing"]= prop_framing_l
    cq_df["eval_precision"]    = eval_prec_l
    cq_df["eval_recall"]       = eval_rec_l
    cq_df["struct_precision"]  = struct_prec_l
    cq_df["struct_recall"]     = struct_rec_l

    return cq_df


def compute_propagation_metrics(
    cq_df: pd.DataFrame,
    systems: List[str],
) -> pd.DataFrame:
    """
    Aggregate propagation metrics for: all / gold / generated / each system.
    Reports both micro (per-CQ) and macro (per-scenario average) metrics,
    plus counts of propagated / not-propagated / newly-introduced framing bias.
    """
    groups: Dict[str, pd.DataFrame] = {
        "all":       cq_df,
        "gold":      cq_df[cq_df["group"] == "gold"],
        "generated": cq_df[cq_df["group"] == "generated"],
    }
    for sys in systems:
        groups[sys] = cq_df[cq_df["system"] == sys]

    def _safe_mean(s: pd.Series) -> float:
        v = s.dropna()
        return round(float(v.mean()), 4) if len(v) > 0 else np.nan

    rows = []
    for gname, gdf in groups.items():
        if gdf.empty:
            continue

        # Keep only rows where a scenario was matched (propagated_framing is not NaN)
        valid = gdf.dropna(subset=["propagated_framing"])

        row: Dict = {
            "group": gname,
            "cqs_with_matched_scenario":  len(valid),
            # Micro metrics (CQ-level)
            "micro_propagated_eval_rate":    _safe_mean(valid["propagated_eval"]),
            "micro_propagated_struct_rate":  _safe_mean(valid["propagated_struct"]),
            "micro_propagated_framing_rate": _safe_mean(valid["propagated_framing"]),
            "micro_mean_eval_precision":     _safe_mean(valid["eval_precision"]),
            "micro_mean_eval_recall":        _safe_mean(valid["eval_recall"]),
            "micro_mean_struct_precision":   _safe_mean(valid["struct_precision"]),
            "micro_mean_struct_recall":      _safe_mean(valid["struct_recall"]),
        }

        # Macro metrics: average per-scenario rates, then average over scenarios
        if len(valid) > 0:
            sc_agg = (
                valid.groupby("scenario_key")
                .agg(
                    sc_prop_eval=    ("propagated_eval",    "mean"),
                    sc_prop_struct=  ("propagated_struct",  "mean"),
                    sc_prop_framing= ("propagated_framing", "mean"),
                    sc_eval_prec=    ("eval_precision",     "mean"),
                    sc_eval_rec=     ("eval_recall",        "mean"),
                    sc_struct_prec=  ("struct_precision",   "mean"),
                    sc_struct_rec=   ("struct_recall",      "mean"),
                )
            )
            row.update({
                "macro_propagated_eval_rate":    round(float(sc_agg["sc_prop_eval"].mean()),    4),
                "macro_propagated_struct_rate":  round(float(sc_agg["sc_prop_struct"].mean()),  4),
                "macro_propagated_framing_rate": round(float(sc_agg["sc_prop_framing"].mean()), 4),
                "macro_mean_eval_precision":     round(float(sc_agg["sc_eval_prec"].dropna().mean()), 4)
                                                 if sc_agg["sc_eval_prec"].notna().any() else np.nan,
                "macro_mean_eval_recall":        round(float(sc_agg["sc_eval_rec"].dropna().mean()), 4)
                                                 if sc_agg["sc_eval_rec"].notna().any() else np.nan,
                "macro_mean_struct_precision":   round(float(sc_agg["sc_struct_prec"].dropna().mean()), 4)
                                                 if sc_agg["sc_struct_prec"].notna().any() else np.nan,
                "macro_mean_struct_recall":      round(float(sc_agg["sc_struct_rec"].dropna().mean()), 4)
                                                 if sc_agg["sc_struct_rec"].notna().any() else np.nan,
            })

        # Categorise CQs: propagated / source-not-propagated / newly-introduced
        n_propagated      = 0
        n_src_not_prop    = 0
        n_newly_introduced = 0

        for _, r in valid.iterrows():
            sc_has_cues = bool(
                (r.get("sc_eval_norm")   or set()) or
                (r.get("sc_struct_norm") or set())
            )
            cq_fb    = bool(r["fb_bool"]) if not _is_nan(r.get("fb_bool")) else False
            prop_f   = r["propagated_framing"]

            if prop_f == 1:
                n_propagated += 1
            elif sc_has_cues and prop_f == 0:
                n_src_not_prop += 1
            if cq_fb and not sc_has_cues and prop_f == 0:
                n_newly_introduced += 1

        row.update({
            "n_propagated_framing_bias":      n_propagated,
            "n_source_bias_not_propagated":   n_src_not_prop,
            "n_newly_introduced_framing":     n_newly_introduced,
        })

        rows.append(row)

    return pd.DataFrame(rows).set_index("group") if rows else pd.DataFrame()


def _is_nan(v) -> bool:
    """Safe NaN check that works for both numeric and non-numeric values."""
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return v is None


# ===========================================================================
# PART 8 — MERGED ROW-LEVEL OUTPUT
# ===========================================================================

def _set_to_str(v) -> str:
    """Convert a set or list to a sorted semicolon-separated string."""
    if isinstance(v, (set, frozenset)):
        return "; ".join(sorted(str(i) for i in v)) if v else ""
    if isinstance(v, list):
        return "; ".join(str(i) for i in v) if v else ""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def build_merged_df(cq_df: pd.DataFrame, sc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-CQ DataFrame combining scenario and CQ annotation columns
    with propagation and inconsistency scores.
    """
    sc_lookup: Dict[str, pd.Series] = {}
    for _, row in sc_df.iterrows():
        k = row["scenario_key"]
        if k:
            sc_lookup[k] = row

    rows = []
    for _, cq_row in cq_df.iterrows():
        sc_row = sc_lookup.get(cq_row["scenario_key"])

        def _sc(field, default=""):
            if sc_row is None or field not in sc_row:
                return default
            v = sc_row[field]
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        rows.append({
            "scenario_key":             cq_row["scenario_key"],
            "scenario_text":            _sc(SCENARIO_TEXT_COL),
            "cq_text":                  cq_row.get(CQ_TEXT_COL, ""),
            "group":                    cq_row["group"],
            "system":                   cq_row["system"],
            # Scenario bias
            "sc_framing_bias":          _sc("fb_bool"),
            "sc_eval_terms_raw":        _set_to_str(_sc("eval_raw",  [])),
            "sc_eval_terms_norm":       _set_to_str(_sc("eval_norm", set())),
            "sc_structural_raw":        _set_to_str(_sc("struct_raw",  [])),
            "sc_structural_norm":       _set_to_str(_sc("struct_norm", set())),
            # CQ bias
            "cq_framing_bias":          cq_row.get("fb_bool"),
            "cq_eval_terms_raw":        _set_to_str(cq_row.get("eval_raw",  [])),
            "cq_eval_terms_norm":       _set_to_str(cq_row.get("eval_norm", set())),
            "cq_structural_raw":        _set_to_str(cq_row.get("struct_raw",  [])),
            "cq_structural_norm":       _set_to_str(cq_row.get("struct_norm", set())),
            "cq_superfluous_raw":       _set_to_str(cq_row.get("superfl_raw",  [])),
            "cq_superfluous_norm":      _set_to_str(cq_row.get("superfl_norm", set())),
            "notes":                    cq_row.get(CQ_NOTES_COL, ""),
            "scenario_motivated":       cq_row.get("scenario_motivated", ""),
            "scenario_motivated_reason":cq_row.get("scenario_motivated_reason", ""),
            # Propagation
            "shared_eval_set":          _set_to_str(cq_row.get("shared_eval",  set())),
            "shared_struct_set":        _set_to_str(cq_row.get("shared_struct", set())),
            "propagated_eval":          cq_row.get("propagated_eval"),
            "propagated_struct":        cq_row.get("propagated_struct"),
            "propagated_framing":       cq_row.get("propagated_framing"),
            "eval_precision":           cq_row.get("eval_precision"),
            "eval_recall":              cq_row.get("eval_recall"),
            "struct_precision":         cq_row.get("struct_precision"),
            "struct_recall":            cq_row.get("struct_recall"),
            # Inconsistency
            "strict_source_inconsistency":   cq_row.get("strict_source_inconsistency"),
            "weighted_source_inconsistency": cq_row.get("weighted_source_inconsistency"),
        })

    return pd.DataFrame(rows)


def build_scenario_summary(
    cq_df: pd.DataFrame,
    sc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one-row-per-scenario summary table comparing gold vs generated CQs.
    """
    def _rate(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        s = df[col].dropna()
        return round(float(s.mean()), 4) if len(s) > 0 else np.nan

    rows = []
    for _, sc_row in sc_df.iterrows():
        sk     = sc_row["scenario_key"]
        sc_cqs = cq_df[cq_df["scenario_key"] == sk]
        gold   = sc_cqs[sc_cqs["group"] == "gold"]
        gen    = sc_cqs[sc_cqs["group"] == "generated"]

        rows.append({
            "scenario_key":                          sk,
            "scenario_text":                         str(sc_row.get(SCENARIO_TEXT_COL, ""))[:120],
            "total_gold_cqs":                        len(gold),
            "total_generated_cqs":                   len(gen),
            "gold_framing_bias_rate":                _rate(gold, "fb_bool"),
            "generated_framing_bias_rate":           _rate(gen,  "fb_bool"),
            "gold_propagation_rate":                 _rate(gold, "propagated_framing"),
            "generated_propagation_rate":            _rate(gen,  "propagated_framing"),
            "gold_strict_inconsistency_rate":        _rate(gold, "strict_source_inconsistency"),
            "generated_strict_inconsistency_rate":   _rate(gen,  "strict_source_inconsistency"),
            "gold_weighted_inconsistency_mean":      _rate(gold, "weighted_source_inconsistency"),
            "generated_weighted_inconsistency_mean": _rate(gen,  "weighted_source_inconsistency"),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# PART 9 — GRAPHICS
# ===========================================================================

PALETTE  = sns.color_palette("Set2")
FIG_DPI  = 200
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    """Save a figure as high-resolution PNG and close it."""
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {path.name}")


def plot_framing_bias_rates(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of framing-bias rates by group (gold / generated / all)."""
    groups = [g for g in ["gold", "generated", "all"] if g in metrics_df.index
              and "prop_framing_bias" in metrics_df.columns]
    if not groups:
        return
    rates = [metrics_df.loc[g, "prop_framing_bias"] for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(groups, rates, color=PALETTE[:len(groups)], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_ylim(0, min(1.05, max(rates) * 1.3 + 0.05) if rates else 1.0)
    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel("Proportion with Framing Bias", fontsize=12)
    ax.set_title("Framing Bias Rate by CQ Group", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    sns.despine(ax=ax)
    _save(fig, "01_framing_bias_rates", out_dir)


def plot_propagation_micro_macro(prop_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: micro vs macro framing-propagation rate by group."""
    groups = [g for g in ["gold", "generated", "all"] if g in prop_df.index]
    if not groups:
        return
    if "micro_propagated_framing_rate" not in prop_df.columns:
        return

    x     = np.arange(len(groups))
    width = 0.32
    micro = [prop_df.loc[g, "micro_propagated_framing_rate"] for g in groups]
    macro = [prop_df.loc[g, "macro_propagated_framing_rate"] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, micro, width, label="Micro (CQ-weighted)",
                color=PALETTE[0], edgecolor="white")
    b2 = ax.bar(x + width / 2, macro, width, label="Macro (scenario-weighted)",
                color=PALETTE[1], edgecolor="white")
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Framing-Bias Propagation Rate", fontsize=12)
    ax.set_title("Framing-Bias Propagation: Micro vs Macro", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    sns.despine(ax=ax)
    _save(fig, "02_propagation_micro_macro", out_dir)


def plot_superfluous_distribution(cq_df: pd.DataFrame, out_dir: Path) -> None:
    """Side-by-side histograms: superfluous-element count per CQ, gold vs generated."""
    if "superfl_count" not in cq_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (lbl, colour) in zip(
        axes,
        [("Gold", PALETTE[0]), ("Generated", PALETTE[1])],
    ):
        sub    = cq_df[cq_df["group"] == lbl.lower()]["superfl_count"]
        max_v  = int(sub.max()) if len(sub) > 0 else 0
        ax.hist(sub, bins=range(0, max_v + 2), align="left",
                color=colour, edgecolor="white", rwidth=0.8)
        ax.set_xlabel("Superfluous Element Count", fontsize=11)
        ax.set_ylabel("Number of CQs", fontsize=11)
        ax.set_title(lbl, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        sns.despine(ax=ax)

    fig.suptitle("Distribution of Superfluous Elements per CQ", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_superfluous_distribution", out_dir)


def plot_cooccurrence_heatmap(cq_df: pd.DataFrame, out_dir: Path) -> None:
    """Symmetric co-occurrence heatmap: framing_bias × eval_terms × struct_assumption."""
    fb   = cq_df["fb_bool"].fillna(False).astype(bool)
    e    = (cq_df["eval_count"]   > 0)
    s    = (cq_df["struct_count"] > 0)

    labels = ["Framing Bias", "Eval Terms", "Struct Assump."]
    matrix = np.array([
        [int(fb.sum()),           int((fb & e).sum()), int((fb & s).sum())],
        [int((fb & e).sum()),     int(e.sum()),         int((e & s).sum())],
        [int((fb & s).sum()),     int((e & s).sum()),   int(s.sum())],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Count"},
    )
    ax.set_title("Bias-Dimension Co-occurrence (All CQs)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_cooccurrence_heatmap", out_dir)


def plot_term_frequencies(
    freq_df: pd.DataFrame,
    title: str,
    fname: str,
    out_dir: Path,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart of top-N normalised term frequencies."""
    if freq_df.empty:
        return
    df = freq_df.head(top_n).sort_values("frequency")
    fig, ax = plt.subplots(figsize=(8, max(3.0, len(df) * 0.38)))
    bars = ax.barh(df["term"], df["frequency"],
                   color=PALETTE[2], edgecolor="white", height=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, fname, out_dir)


def plot_micro_macro_fb_generated(
    per_sc_results: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Compare micro vs macro framing-bias rate for generated CQs."""
    gen_sc = per_sc_results.get("generated", pd.DataFrame())
    if gen_sc.empty or "fb_rate" not in gen_sc.columns:
        return

    macro_rate = float(gen_sc["fb_rate"].mean())
    micro_rate = (
        float(metrics_df.loc["generated", "prop_framing_bias"])
        if "generated" in metrics_df.index else np.nan
    )
    if _is_nan(micro_rate):
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Micro\n(CQ-weighted)", "Macro\n(Scenario-weighted)"],
        [micro_rate, macro_rate],
        color=PALETTE[:2], edgecolor="white",
    )
    for bar, val in zip(ax.patches, [micro_rate, macro_rate]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{val:.2%}",
            ha="center", fontsize=11,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Framing Bias Rate", fontsize=12)
    ax.set_title("Generated CQs: Micro vs Macro\nFraming Bias Rate", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "05_micro_macro_generated_fb", out_dir)


def plot_generated_cq_count_dist(
    per_sc_results: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Bar chart of how many scenarios have 1 / 2 / 3+ generated CQs."""
    gen_df = per_sc_results.get("generated", pd.DataFrame())
    if gen_df.empty or "cq_count" not in gen_df.columns:
        return
    cnt = gen_df["cq_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cnt.index.astype(str), cnt.values, color=PALETTE[3], edgecolor="white")
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(int(bar.get_height())),
            ha="center", fontsize=10,
        )
    ax.set_xlabel("Number of Generated CQs per Scenario", fontsize=11)
    ax.set_ylabel("Number of Scenarios",                  fontsize=11)
    ax.set_title("Distribution of Generated CQ Count per Scenario",
                 fontsize=12, fontweight="bold")
    sns.despine(ax=ax)
    _save(fig, "06_generated_cq_count_dist", out_dir)


def plot_inconsistency_by_motivated(cq_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: mean weighted inconsistency by scenario_motivated category."""
    if ("scenario_motivated" not in cq_df.columns
            or "weighted_source_inconsistency" not in cq_df.columns):
        return
    sub = cq_df[cq_df["superfl_count"] > 0].copy()
    if sub.empty:
        return
    grp = sub.groupby("scenario_motivated")["weighted_source_inconsistency"].mean()
    if grp.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(grp.index, grp.values, color=PALETTE[:len(grp)], edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xlabel("scenario_motivated", fontsize=11)
    ax.set_ylabel("Mean Weighted Source Inconsistency", fontsize=11)
    ax.set_title("Weighted Inconsistency by Scenario-Motivated Category",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    sns.despine(ax=ax)
    plt.tight_layout()
    _save(fig, "07_weighted_inconsistency_by_motivated", out_dir)


def plot_scenario_summary_heatmap(
    sc_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap of per-scenario gold vs generated framing-bias and propagation rates."""
    cols = [
        "gold_framing_bias_rate",
        "generated_framing_bias_rate",
        "gold_propagation_rate",
        "generated_propagation_rate",
    ]
    cols = [c for c in cols if c in sc_summary.columns]
    if not cols or sc_summary.empty:
        return

    numeric = sc_summary[cols].astype(float)
    labels  = sc_summary["scenario_key"].str[:40].tolist()

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    sns.heatmap(
        numeric, annot=True, fmt=".2f", cmap="Blues",
        yticklabels=labels, xticklabels=[c.replace("_", " ") for c in cols],
        linewidths=0.4, ax=ax, cbar_kws={"label": "Rate"},
        vmin=0, vmax=1,
    )
    ax.set_title("Per-Scenario Framing Bias & Propagation Rates",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "08_scenario_summary_heatmap", out_dir)


def plot_propagation_categories(prop_df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar chart of propagation categories per group."""
    cols = [
        "n_propagated_framing_bias",
        "n_source_bias_not_propagated",
        "n_newly_introduced_framing",
    ]
    cols = [c for c in cols if c in prop_df.columns]
    if not cols:
        return

    groups = [g for g in ["gold", "generated", "all"] if g in prop_df.index]
    data   = prop_df.loc[groups, cols].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom  = np.zeros(len(groups))
    nice    = ["Propagated", "Source not Propagated", "Newly Introduced"]

    for col, lbl, colour in zip(cols, nice, PALETTE[:3]):
        vals = data[col].values
        ax.bar(groups, vals, bottom=bottom, label=lbl, color=colour, edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, str(int(v)), ha="center",
                        va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_xlabel("Group", fontsize=12)
    ax.set_ylabel("Number of CQs", fontsize=12)
    ax.set_title("Framing-Bias Propagation Categories by Group",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    _save(fig, "09_propagation_categories", out_dir)


# ===========================================================================
# PART 10 — OUTPUT FILES
# ===========================================================================

def write_outputs(
    out_dir: Path,
    sc_metrics:   Dict,
    sc_eval_freq: pd.DataFrame,
    sc_struct_freq: pd.DataFrame,
    cq_metrics_df:  pd.DataFrame,
    cq_freq_tables: Dict,
    per_sc_results: Dict[str, pd.DataFrame],
    micro_macro_agg: Dict,
    prop_metrics_df: pd.DataFrame,
    merged_df:   pd.DataFrame,
    sc_summary:  pd.DataFrame,
) -> None:
    """Write all output CSV, JSON, and summary files to out_dir."""

    # 1. Scenario metrics
    pd.DataFrame([sc_metrics]).to_csv(out_dir / "scenario_metrics.csv", index=False)

    # 2. Scenario eval-term frequencies
    sc_eval_freq.to_csv(out_dir / "scenario_eval_term_frequencies.csv", index=False)

    # 3. Scenario structural-assumption frequencies
    sc_struct_freq.to_csv(out_dir / "scenario_structural_assumption_frequencies.csv", index=False)

    # 4. CQ metrics by group (micro-level / CQ-weighted)
    cq_metrics_df.to_csv(out_dir / "cq_metrics_by_group.csv")

    # 5. Micro vs macro comparison
    mm_rows = []
    for grp_lbl, m in micro_macro_agg.get("macro", {}).items():
        r = {"group": grp_lbl, **m}
        mm_rows.append(r)
    if mm_rows:
        pd.DataFrame(mm_rows).to_csv(out_dir / "cq_metrics_micro_vs_macro.csv", index=False)

    # 6. Scenario-level CQ summary
    sc_summary.to_csv(out_dir / "scenario_level_cq_summary.csv", index=False)

    # 7–9. Frequency tables by group
    fname_map = {
        "eval":   "cq_eval_term_frequencies_by_group.csv",
        "struct": "cq_structural_assumption_frequencies_by_group.csv",
        "superfl":"cq_superfluous_frequencies_by_group.csv",
    }
    for key, fname in fname_map.items():
        parts = []
        for gname, fdf in cq_freq_tables.get(key, {}).items():
            if not fdf.empty:
                fdf = fdf.copy()
                fdf["group"] = gname
                parts.append(fdf)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out_dir / fname, index=False)

    # 10. Propagation metrics by group
    if not prop_metrics_df.empty:
        prop_metrics_df.to_csv(out_dir / "propagation_metrics_by_group.csv")

    # 11. Merged row-level analysis
    merged_df.to_csv(out_dir / "merged_cq_scenario_analysis.csv", index=False)

    # 12. Summary JSON report
    report: Dict = {
        "generated_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "scenario_metrics": sc_metrics,
        "micro_macro_aggregates": micro_macro_agg.get("macro", {}),
        "generated_cq_exposure": micro_macro_agg.get("exposure", {}),
    }
    if not cq_metrics_df.empty:
        report["cq_metrics_by_group"] = cq_metrics_df.to_dict(orient="index")
    if not prop_metrics_df.empty:
        report["propagation_metrics_by_group"] = prop_metrics_df.to_dict(orient="index")

    with open(out_dir / "summary_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"All {len(list(out_dir.glob('*')))} output files written to {out_dir}/")


# ===========================================================================
# PART 11 — CONSOLE SUMMARY
# ===========================================================================

def print_summary(
    sc_metrics:      Dict,
    cq_metrics_df:   pd.DataFrame,
    prop_metrics_df: pd.DataFrame,
    micro_macro_agg: Dict,
) -> None:
    """Print a human-readable summary to stdout."""

    SEP = "=" * 62
    DIV = "─" * 42

    print(f"\n{SEP}")
    print("  BIAS ANALYSIS — SUMMARY REPORT")
    print(SEP)

    # Scenarios
    print(f"\n{DIV}")
    print("  SCENARIOS")
    print(DIV)
    n_sc = sc_metrics.get("total_scenarios", "N/A")
    fb_r = sc_metrics.get("prop_framing_bias", 0) * 100
    print(f"  Total distinct scenarios:       {n_sc}")
    print(f"  Framing bias (count):           {sc_metrics.get('n_framing_bias','N/A')}")
    print(f"  Framing bias rate:              {fb_r:.1f}%")
    print(f"  With evaluative terms:          {sc_metrics.get('n_with_eval_terms','N/A')}")
    print(f"  With structural assumptions:    {sc_metrics.get('n_with_structural_assumption','N/A')}")
    print(f"  Mean eval terms/scenario:       {sc_metrics.get('mean_eval_per_scenario', 0):.3f}")
    print(f"  Mean struct assumptions/sc:     {sc_metrics.get('mean_struct_per_scenario', 0):.3f}")

    # CQ counts
    if not cq_metrics_df.empty:
        print(f"\n{DIV}")
        print("  CQ COUNTS")
        print(DIV)
        for grp in ("all", "gold", "generated"):
            if grp in cq_metrics_df.index:
                n  = int(cq_metrics_df.loc[grp, "total_cqs"])
                ds = int(cq_metrics_df.loc[grp, "distinct_scenarios"])
                print(f"  {grp.capitalize():12s}:  {n:3d} CQs  ({ds} distinct scenarios)")

        print(f"\n{DIV}")
        print("  FRAMING BIAS RATES")
        print(DIV)
        for grp in ("all", "gold", "generated"):
            if grp in cq_metrics_df.index:
                rate = cq_metrics_df.loc[grp, "prop_framing_bias"] * 100
                print(f"  {grp.capitalize():12s}:  {rate:.1f}%")

        if "strict_inconsistency_rate" in cq_metrics_df.columns:
            print(f"\n{DIV}")
            print("  SOURCE INCONSISTENCY (CQ-weighted)")
            print(DIV)
            for grp in ("all", "gold", "generated"):
                if grp in cq_metrics_df.index:
                    strict   = cq_metrics_df.loc[grp, "strict_inconsistency_rate"]   * 100
                    weighted = cq_metrics_df.loc[grp, "weighted_inconsistency_mean"]
                    print(f"  {grp.capitalize():12s}:  strict={strict:.1f}%   "
                          f"weighted_mean={weighted:.3f}")

        print(f"\n{DIV}")
        print("  CQs PER SCENARIO — generated")
        print(DIV)
        if "generated" in cq_metrics_df.index:
            mean_  = cq_metrics_df.loc["generated", "mean_cqs_per_scenario"]
            med_   = cq_metrics_df.loc["generated", "median_cqs_per_scenario"]
            max_   = cq_metrics_df.loc["generated", "max_cqs_per_scenario"]
            print(f"  Mean:    {mean_:.2f}")
            print(f"  Median:  {med_:.2f}")
            print(f"  Max:     {max_}")

    # Propagation
    if not prop_metrics_df.empty and "micro_propagated_framing_rate" in prop_metrics_df.columns:
        print(f"\n{DIV}")
        print("  FRAMING-BIAS PROPAGATION (micro | macro)")
        print(DIV)
        for grp in ("all", "gold", "generated"):
            if grp in prop_metrics_df.index:
                mi = prop_metrics_df.loc[grp, "micro_propagated_framing_rate"]
                ma = prop_metrics_df.loc[grp, "macro_propagated_framing_rate"]
                print(f"  {grp.capitalize():12s}:  micro={mi*100:.1f}%   macro={ma*100:.1f}%")

    # Macro aggregates
    mm = micro_macro_agg.get("macro", {})
    if mm:
        print(f"\n{DIV}")
        print("  MACRO METRICS (scenario-weighted)")
        print(DIV)
        for lbl in ("generated", "gold"):
            d = mm.get(lbl, {})
            if d:
                print(f"  [{lbl}]")
                for k, v in d.items():
                    print(f"    {k}: {v}")

    # Exposure
    exp = micro_macro_agg.get("exposure", {})
    if exp:
        print(f"\n{DIV}")
        print("  GENERATED CQ EXPOSURE (scenarios × count)")
        print(DIV)
        print(f"  Scenarios with 1 CQ :   {exp.get('scenarios_with_1_cq',  0)}")
        print(f"  Scenarios with 2 CQs:   {exp.get('scenarios_with_2_cqs', 0)}")
        print(f"  Scenarios with 3+ CQs:  {exp.get('scenarios_with_3plus_cqs', 0)}")

    print(f"\n{SEP}\n")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── PART 1: Load ──────────────────────────────────────────────────────────
    sc_raw = load_file(SCENARIO_FILE, SCENARIO_SHEET)
    cq_raw = load_file(CQ_FILE,       CQ_SHEET)

    validate_columns(
        sc_raw,
        required=[SCENARIO_TEXT_COL],
        optional=[SCENARIO_FRAMING_BIAS_COL, SCENARIO_EVAL_TERMS_COL,
                  SCENARIO_STRUCTURAL_COL, SCENARIO_NOTES_COL],
        label="scenario file",
    )
    validate_columns(
        cq_raw,
        required=[CQ_SCENARIO_TEXT_COL, CQ_TEXT_COL, CQ_GROUP_COL],
        optional=[CQ_ID_COL, CQ_FRAMING_BIAS_COL, CQ_EVAL_TERMS_COL,
                  CQ_STRUCTURAL_COL, CQ_SUPERFLUOUS_COL, CQ_NOTES_COL],
        label="CQ file",
    )

    # ── Scenario preprocessing ────────────────────────────────────────────────
    # Keep only rows with non-empty scenario text
    sc_df = sc_raw[sc_raw[SCENARIO_TEXT_COL].notna()].copy()
    sc_df = sc_df[sc_df[SCENARIO_TEXT_COL].str.strip() != ""].copy()

    # Build matching keys
    sc_df["scenario_key"] = sc_df[SCENARIO_TEXT_COL].apply(make_scenario_key)

    # Deduplicate: if the same key appears in multiple rows, prefer the annotated one
    if SCENARIO_FRAMING_BIAS_COL in sc_df.columns:
        sc_df["_has_ann"] = sc_df[SCENARIO_FRAMING_BIAS_COL].notna()
    else:
        sc_df["_has_ann"] = False

    sc_df = (
        sc_df
        .sort_values("_has_ann", ascending=False)
        .drop_duplicates(subset=["scenario_key"], keep="first")
        .drop(columns=["_has_ann"])
        .reset_index(drop=True)
    )

    # Apply exclusion list (matched against normalised key prefix)
    if EXCLUDED_SCENARIO_PREFIXES:
        before = len(sc_df)
        mask = sc_df["scenario_key"].apply(
            lambda k: not any(k.startswith(p.lower().strip()) for p in EXCLUDED_SCENARIO_PREFIXES)
        )
        sc_df = sc_df[mask].reset_index(drop=True)
        excluded_n = before - len(sc_df)
        if excluded_n:
            log.info(f"Excluded {excluded_n} scenario(s) via EXCLUDED_SCENARIO_PREFIXES.")

    log.info(f"Distinct scenarios after deduplication: {len(sc_df)}")

    # Treat empty framing_bias as No (user instruction)
    if SCENARIO_FRAMING_BIAS_COL in sc_df.columns:
        sc_df[SCENARIO_FRAMING_BIAS_COL] = sc_df[SCENARIO_FRAMING_BIAS_COL].fillna("No")

    # Enrich scenario annotations
    sc_df = enrich_annotations(
        sc_df,
        fb_col     = SCENARIO_FRAMING_BIAS_COL if SCENARIO_FRAMING_BIAS_COL in sc_df.columns else None,
        eval_col   = SCENARIO_EVAL_TERMS_COL   if SCENARIO_EVAL_TERMS_COL   in sc_df.columns else None,
        struct_col = SCENARIO_STRUCTURAL_COL   if SCENARIO_STRUCTURAL_COL   in sc_df.columns else None,
    )
    # Scenarios: treat None fb as False
    sc_df["fb_bool"] = sc_df["fb_bool"].fillna(False)

    # ── CQ preprocessing ──────────────────────────────────────────────────────
    cq_df = cq_raw.copy()
    cq_df["scenario_key"] = cq_df[CQ_SCENARIO_TEXT_COL].apply(make_scenario_key)

    # group column: "gold" vs "generated"
    cq_df["group"] = cq_df[CQ_GROUP_COL].apply(
        lambda v: "gold" if str(v).strip().lower() == GOLD_LABEL.lower() else "generated"
    )
    # system column: original Source value
    cq_df["system"] = cq_df[CQ_SYSTEM_COL].apply(
        lambda v: str(v).strip() if not (v is None or (isinstance(v, float) and np.isnan(v))) else "unknown"
    )
    # Drop CQs whose scenario key matches the exclusion list
    if EXCLUDED_SCENARIO_PREFIXES:
        cq_mask = cq_df["scenario_key"].apply(
            lambda k: not any(k.startswith(p.lower().strip()) for p in EXCLUDED_SCENARIO_PREFIXES)
        )
        n_before = len(cq_df)
        cq_df = cq_df[cq_mask].reset_index(drop=True)
        if len(cq_df) < n_before:
            log.info(f"Excluded {n_before - len(cq_df)} CQ row(s) matching EXCLUDED_SCENARIO_PREFIXES.")

    systems = sorted(set(cq_df["system"].unique()) - {GOLD_LABEL})
    log.info(f"CQ groups: gold={int((cq_df['group']=='gold').sum())}  "
             f"generated={int((cq_df['group']=='generated').sum())}")
    log.info(f"Systems/models (non-gold): {systems}")

    unmatched = set(cq_df["scenario_key"].unique()) - set(sc_df["scenario_key"].unique())
    if unmatched:
        log.warning(
            f"{len(unmatched)} CQ scenario key(s) have no matching scenario annotation "
            f"— propagation metrics will be NaN for those CQs."
        )

    # Enrich CQ annotations
    cq_df = enrich_annotations(
        cq_df,
        fb_col     = CQ_FRAMING_BIAS_COL if CQ_FRAMING_BIAS_COL in cq_df.columns else None,
        eval_col   = CQ_EVAL_TERMS_COL   if CQ_EVAL_TERMS_COL   in cq_df.columns else None,
        struct_col = CQ_STRUCTURAL_COL   if CQ_STRUCTURAL_COL   in cq_df.columns else None,
        superfl_col= CQ_SUPERFLUOUS_COL  if CQ_SUPERFLUOUS_COL  in cq_df.columns else None,
    )

    # ── PART 3: scenario_motivated via Claude API ─────────────────────────────
    cq_df = add_scenario_motivated(
        cq_df,
        notes_col     = CQ_NOTES_COL       if CQ_NOTES_COL       in cq_df.columns else "__missing__",
        superfluous_col= CQ_SUPERFLUOUS_COL if CQ_SUPERFLUOUS_COL in cq_df.columns else None,
    )

    # Inconsistency scores require scenario_motivated to be set
    cq_df = add_inconsistency_scores(cq_df)

    # ── PART 7: Propagation (must run before per-scenario counts use it) ──────
    cq_df = compute_propagation(cq_df, sc_df)

    # ── PART 4: Scenario metrics ──────────────────────────────────────────────
    sc_metrics, sc_eval_freq, sc_struct_freq = compute_scenario_metrics(sc_df)

    # ── PART 5: CQ metrics by group ───────────────────────────────────────────
    cq_metrics_df, cq_freq_tables = compute_cq_metrics(cq_df, systems)

    # ── PART 6: Per-scenario counts, micro vs macro ───────────────────────────
    per_sc_results, micro_macro_agg = compute_per_scenario_counts(cq_df)

    # ── Propagation metrics by group ──────────────────────────────────────────
    prop_metrics_df = compute_propagation_metrics(cq_df, systems)

    # ── PART 8: Merged outputs ────────────────────────────────────────────────
    merged_df   = build_merged_df(cq_df, sc_df)
    sc_summary  = build_scenario_summary(cq_df, sc_df)

    # ── PART 9: Graphics ──────────────────────────────────────────────────────
    plot_framing_bias_rates(cq_metrics_df, out_dir)
    plot_propagation_micro_macro(prop_metrics_df, out_dir)
    plot_superfluous_distribution(cq_df, out_dir)
    plot_cooccurrence_heatmap(cq_df, out_dir)
    plot_micro_macro_fb_generated(per_sc_results, cq_metrics_df, out_dir)
    plot_generated_cq_count_dist(per_sc_results, out_dir)
    plot_inconsistency_by_motivated(cq_df, out_dir)
    plot_scenario_summary_heatmap(sc_summary, out_dir)
    plot_propagation_categories(prop_metrics_df, out_dir)

    # Term-frequency charts
    plot_term_frequencies(
        sc_eval_freq,
        "Evaluative Terms in Scenarios",
        "10_scenario_eval_frequencies", out_dir,
    )
    plot_term_frequencies(
        sc_struct_freq,
        "Structural Assumptions in Scenarios",
        "11_scenario_struct_frequencies", out_dir,
    )
    for grp in ("all", "gold", "generated"):
        fdf = cq_freq_tables.get("eval", {}).get(grp, pd.DataFrame())
        plot_term_frequencies(
            fdf, f"Evaluative Terms — {grp.capitalize()} CQs",
            f"12_eval_freq_{grp}", out_dir,
        )
        fdf = cq_freq_tables.get("struct", {}).get(grp, pd.DataFrame())
        plot_term_frequencies(
            fdf, f"Structural Assumptions — {grp.capitalize()} CQs",
            f"13_struct_freq_{grp}", out_dir,
        )
        fdf = cq_freq_tables.get("superfl", {}).get(grp, pd.DataFrame())
        plot_term_frequencies(
            fdf, f"Superfluous Elements — {grp.capitalize()} CQs",
            f"14_superfl_freq_{grp}", out_dir,
        )

    # ── PART 10: Write files ──────────────────────────────────────────────────
    write_outputs(
        out_dir,
        sc_metrics, sc_eval_freq, sc_struct_freq,
        cq_metrics_df, cq_freq_tables,
        per_sc_results, micro_macro_agg,
        prop_metrics_df, merged_df, sc_summary,
    )

    # ── PART 11: Console summary ──────────────────────────────────────────────
    print_summary(sc_metrics, cq_metrics_df, prop_metrics_df, micro_macro_agg)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()


# ===========================================================================
# METHODOLOGICAL NOTES
# ===========================================================================
#
# WEIGHTED SOURCE-INCONSISTENCY
#   A CQ row with superfluous elements gets a score of:
#   0.0 — no superfluous elements at all (no inconsistency).
#   0.5 — has superfluous elements AND scenario_motivated == "yes"
#          (element is present but justifiable from the source scenario context).
#   1.0 — has superfluous elements AND scenario_motivated in {"no","uncertain"}
#          (element is genuinely unexplained by the scenario, or unclear).
#   This allows a nuanced aggregation: rows where superfluous content is
#   scenario-attributable contribute only half the inconsistency penalty.
#
# FRAMING-BIAS PROPAGATION
#   For each CQ linked to a scenario (matched via normalised 100-char key):
#     shared_eval   = eval_norm(scenario) ∩ eval_norm(CQ)
#     shared_struct = struct_norm(scenario) ∩ struct_norm(CQ)
#   propagated_eval   = 1 iff shared_eval   is non-empty
#   propagated_struct = 1 iff shared_struct is non-empty
#   propagated_framing= 1 iff either is non-empty
#   Matching is exact (after lowercase + whitespace normalisation).
#   No fuzzy matching, cosine similarity, or embedding-based expansion is used.
#   Multi-word expressions (e.g. "non elite", "level of confidence") are preserved.
#
# MICRO vs MACRO METRICS
#   Micro: every CQ counts equally.  A scenario with 19 CQs has 19× the weight
#          of a scenario with 1 CQ.  Standard group-level mean = micro metric.
#   Macro: every scenario counts equally, regardless of how many CQs it has.
#          Per-scenario rates are computed first, then averaged across scenarios.
#   Comparison reveals whether results are dominated by high-volume scenarios.
#
# NUMBER OF GENERATED CQs PER SCENARIO
#   Counted in compute_per_scenario_counts().  The exposure table shows how many
#   scenarios have 1 / 2 / 3+ generated CQs.  All macro aggregates (framing-bias
#   rate, inconsistency rate, propagation rate) are weighted at the scenario level
#   so that scenarios with many CQs do not disproportionately drive the summary.
#
# ANNOTATION FORMAT ASSUMPTIONS
#   • Files are CSV or Excel (.xlsx/.xls/.xlsm/.ods).
#   • Boolean annotations: Yes/No, TRUE/FALSE, 1/0, or empty (empty → False for
#     scenarios per user instruction; NaN-propagated for CQs).
#   • Annotation cells may carry a leading count prefix ("2: term1, term2" or
#     "1 (renovated)") which is stripped before splitting into items.
#   • Item separators: comma, semicolon, pipe, slash, newline, plus-sign.
#   • Null placeholders: -, 0, nan, none, n/a are treated as empty annotations.
#   • Scenario matching uses the first 100 normalised characters (lowercase,
#     whitespace collapsed, sentence-ending punctuation stripped).
#   • CQs whose scenario text does not match any scenario-file entry are kept in
#     all CQ-level metrics but receive NaN for propagation columns.
