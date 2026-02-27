# Deprecated — v1 Pipeline

This directory contains the original RLM-based test set creation pipeline. It has been superseded by the `pipeline/` module.

## Why deprecated

The v1 pipeline used an RLM (Recursive Language Model) REPL agent to navigate documents and extract passages. While functional, it had several issues:

- **Slow**: Multiple REPL iterations per question (many LLM calls)
- **Unreliable passage extraction**: The RLM sometimes paraphrased instead of copying verbatim
- **Expensive**: High token usage from recursive document navigation

## What replaced it

The `pipeline/` module uses:
- **BM25 search index** for deterministic passage retrieval (no LLM needed)
- **Regex-based entity extraction** for intelligent section scoring
- **Single LLM call per question** (generation only)

## Contents

- `v1_src/` — Original pipeline code (`config.py`, `rlm_integration.py`, `test_set_creator.py`, `question_generators.py`)
- `v1_main.py` — Original entry point
- `rlm-minimal-main/` — The RLM framework dependency
- `test_rlm_question.py` — v1 smoke test
