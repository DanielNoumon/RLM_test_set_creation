# Test Set Creator

Automated golden Q&A test set generation for RAG applications. Parses PDF/TXT/MD documents, selects passages using BM25 + entity extraction, and generates question-answer pairs via a single LLM call per question.

## Architecture

```
pipeline/          ← active pipeline (Parse → Index → Select → Generate → Validate)
deprecated/        ← v1 RLM-based pipeline (preserved for reference)
test-set-viewer/   ← web UI for reviewing generated test sets
data/              ← input documents + output test sets
```

See `pipeline/WORKFLOW.md` for a detailed pipeline diagram and phase descriptions.

## Question Types (15 enabled / 20 defined)

- **Direct Lookup** — exact fact retrieval
- **Paraphrase Lookup** — rephrased fact retrieval
- **Specific Jargon** — domain term definitions
- **Multi-hop Within** — combine info from same document
- **Multi-hop Between** — combine info across documents
- **Temporal** — document versioning / recency
- **Pinpointing/Quoting** — source location identification
- **Long Context Synthesis** — structural/counting across sections
- **Needle in Haystack** — hidden detail retrieval
- **Ambiguous Questions** — intentionally vague wording
- **Lists Extraction** — items from bullet/numbered lists
- **Hallucination Test** — unanswerable questions (BM25-validated)
- **Adversarial/Aggro** — aggressive tone with de-escalation
- **Prompt Injection** — jailbreak resistance
- **Multi-turn Followup** — conversational memory

## Installation

```bash
conda create -n rlm_test_set_creation python=3.11
conda activate rlm_test_set_creation
pip install -r requirements.txt
```

## Configuration

Set up `.env`:
```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT_GPT5=your_endpoint
```

Edit question types and counts in `main.py`:
```python
QUESTION_TYPES = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="easy"
    ),
    # ...
}
```

## Usage

```bash
python main.py
```

### Input

Place documents in `data/files_for_test_set/`:
- `.pdf`, `.txt`, `.md` files

### Output

Test sets saved to `data/test_sets/{corpus_name}/v2_*.json` with:
- Questions with golden answers and context
- Per-question metadata (source docs, pages, match ratios)
- MLflow metrics (SQLite: `mlflow.db`)

## License

MIT License — see LICENSE file for details.
