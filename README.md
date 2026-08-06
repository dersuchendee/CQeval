# Evaluating Competency Questions: Measuring Perspectivisation from Requirement Sources

This repository contains the full pipeline for generating, annotating, and analysing competency questions (CQs) produced by two LLM-based ontology engineering assistants — OntoChat and NeOn-GPT — across four ontology projects (Polifonia, IntelligentBathrooms, WHOW, CVN). The analysis focuses on two dimensions: **source additions** (concepts in a CQ not grounded in the source scenario) and **appraisal terms** (normative or non-neutral language).

---

## Setup

```bash
pip install -r requirements.txt
```

Set your API key (required for all `annotate_*.py` and `generate_*.py` scripts):

```bash
export apikey=your-key-here
```

All scripts use paths relative to the repository root via `BASE_DIR = pathlib.Path(__file__).parent`. No manual path configuration is needed as long as you run the scripts from within the cloned repository.

---

## Pipeline

The pipeline runs in the following order:

### 1. CQ Generation

Generate CQs from scenarios using each system:

```bash
python generate_polifonia_cqs.py
python generate_neongpt_cqs.py
python generate_ontochat_ib_whow_cvn.py
python generate_neongpt_ib_whow_cvn.py
```

### 2. Preprocessing

Flatten and clean the generated outputs:

```bash
python disaggregate_ontochat.py
python disaggregate_neongpt.py
python fix_scenarios.py
```

### 3. LLM Annotation

Annotate each CQ for superfluous elements, evaluative terms, SE justification, and SE type:

```bash
python annotate_superfluous.py
python annotate_polifonia_superfluous.py
python annotate_eval_terms.py
python annotate_polifonia_eval_terms.py
python annotate_se_justified.py
python annotate_se_types.py
```

Annotation results are cached in `results/` as JSON files so that re-runs do not repeat API calls.

### 4. Analysis

Run the main analysis and deviation detection:

```bash
python bias_analysis.py
python analyze_superfluous_deviation.py
python analyze_eval_terms_deviation.py
```

### 5. Human Annotation Subsets

Build subsets for human annotation:

```bash
python build_annotation_subsets.py       # random stratified subsets
python build_flag_annotation_subsets.py  # subsets drawn from flagged cases
python build_eval_subset.py              # evaluation subset for LLM annotation quality
```

Output files in `results/` ending in `_annotator.csv` are intended for distribution to annotators (no gold labels); files ending in `_with_gold.csv` are the researcher reference copies.

---

## Data

### LLM-annotated files (`data/annotated_*.csv`)

Each file contains one CQ per row with columns:

| Column | Description |
|--------|-------------|
| `Scenario` / `scenario` | Source scenario text |
| `Question` / `CQ` | Competency question |
| `Source` / `System` | Generating system (OntoChat, GPT4o, Qwen3, Gold) |
| `superfluous_elements` | Elements in the CQ not present in the scenario |
| `eval_terms` | Evaluative or non-neutral terms in the CQ |
| `source_consistency` | Human label: is the CQ consistent with the scenario? (Polifonia only) |

### Human annotation files (`data/human_annotations/`)

Annotation subsets distributed to human annotators for inter-annotator agreement (IAA) assessment. Each file covers one annotation subset and one annotator. Columns: `dataset`, `scenario`, `cq`, `source_deviation`, `framing`, `notes`.

### Results (`results/`)

Computed outputs including per-group metrics, IQR-based deviation flags, SE type distributions, and annotation subsets.

---

## Inter-annotator agreement

IAA was assessed using Cohen's κ and Gwet's AC1 across four annotator pairs covering three annotation subsets (n = 43 CQ judgements pooled). Pooled agreement: κ = 0.610 / AC1 = 0.783 for source deviation; κ = 0.724 / AC1 = 0.942 for framing. LLM–human agreement closely matched human–human levels (source deviation κ = 0.563 / AC1 = 0.796; framing κ = 0.614 / AC1 = 0.878).

---

## Guidelines

The annotation guidelines are available in: [`GUIDELINES.md`](GUIDELINES.md)

---

## Examples of justified and unjustified source additions

These examples are intended to illustrate the operational distinction between *justified* and *unjustified* additions used during annotation. They should not be interpreted as a quality assessment of the competency questions themselves.

| User Story | Competency Question | Source Addition | Justified? | Annotator rationale |
|------------|--------------------|---------------|------------|---------------------|
| There are several actors involved in a construction use case/circular value flow, each holding some roles in a certain material flow. | What are the key processes involved in the recycling of a material (e.g., "concrete")? | recycling | Yes | reasonably inferred from circular value flow |
| There are several actors involved in a construction use case/circular value flow, each holding some roles in a certain material flow. | How does the ontology model the alignment with industry standards (e.g., EN 15804)? | industry standards | No |  not supported by the user story  |
| Based on the fact that I have the evening off, that it is going to rain, and that I like to go to the movies on my free-time, the system finds a film that I could see tonight. | Who is the director of a specific movie? | specific movie | Yes | immediate specification of the film identified by the system (film found tonight) |
| Based on the fact that I have the evening off, that it is going to rain, and that I like to go to the movies on my free-time, the system finds a film that I could see tonight. | Who is the director of a specific movie? | director | No | not supported by the user story |

### Interpretation

Source additions are identified by their absence as explicit expressions in
the User Story. Therefore, synonyms, paraphrases, and coreferential reformulation of content explicitly stated in the User Story may still be annotated as source additions and subsequently marked as
justified.

A source addition is considered **justified** when it is a synonym or paraphrase
of an expression in the User Story, or an immediate specification directly
licensed by the scenario text. The assessment relies on the User Story alone
and does not use external domain knowledge.

A source addition is considered **unjustified** when the User Story provides no
textual support for it.

Importantly, unjustified additions do **not** necessarily indicate low-quality competency questions. Domain experts and knowledge engineers often introduce additional concepts based on domain knowledge, modelling experience, or implicit assumptions. For this reason, source additions are treated as **deviations from the user story** rather than direct indicators of CQ quality.

---

## License

Apache 2.0
