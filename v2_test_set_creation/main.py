"""
v2 Test Set Creator — Entry Point.

Index-first architecture: Parse → Index → Select → Generate → Validate.
Only uses LLM for Q+A generation (1 call per question).
"""
import os
import sys
from pathlib import Path

# Ensure the v2 package is importable when running this file directly
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import time
from datetime import datetime

from dotenv import load_dotenv
import mlflow

from v2_test_set_creation.config import (
    TestSetConfig, LLMConfig, QuestionType, QuestionConfig,
)
from v2_test_set_creation.pipeline import Pipeline

# Load environment variables
load_dotenv()

# Configure MLflow tracking
mlflow.openai.autolog()
mlflow_db = _THIS_DIR / "mlflow_v2.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.set_experiment("v2-test-set-creation")
mlflow.disable_system_metrics_logging()
print(f"MLflow tracking enabled (SQLite): {mlflow_db}")

# Set Azure OpenAI environment variables
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT5")
azure_api_version = os.getenv(
    "AZURE_OPENAI_API_VERSION_GPT5", "2024-12-01-preview"
)

# Also set the generic env vars for the LLM client
if azure_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = azure_key
if azure_endpoint and not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
if azure_api_version and not os.getenv("AZURE_OPENAI_API_VERSION"):
    os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version


def build_config(
    model: str = "gpt-5",
    corpus_name: str = "DSL_corpus",
    input_documents_path: str = None,
    output_path: str = None,
    question_types: dict = None,
    random_seed: int = 42,
) -> TestSetConfig:
    """Build a TestSetConfig with sensible defaults.

    Override any parameter to experiment with different settings.
    """
    project_root = str(_THIS_DIR.parent)

    if input_documents_path is None:
        input_documents_path = os.path.join(
            project_root, "data", "files_for_test_set"
        )
    if output_path is None:
        output_path = os.path.join(
            project_root, "data", "test_sets"
        )
    if question_types is None:
        question_types = ALL_TYPES_DEFAULT

    return TestSetConfig(
        llm=LLMConfig(
            api_key=azure_key,
            model=model,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        ),
        corpus_name=corpus_name,
        input_documents_path=input_documents_path,
        output_path=output_path,
        question_types=question_types,
        random_seed=random_seed,
    )


def run_pipeline(config: TestSetConfig):
    """Run the pipeline with MLflow tracking."""
    enabled = config.get_enabled_question_types()
    total_q = config.get_total_enabled_questions()

    run_name = (
        f"v2_{config.corpus_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    mlflow.start_run(run_name=run_name)

    # Log configuration params
    mlflow.log_param("pipeline_version", "v2")
    mlflow.log_param("model", config.llm.model)
    mlflow.log_param("corpus_name", config.corpus_name)
    mlflow.log_param("questions_requested", total_q)
    mlflow.log_param("random_seed", config.random_seed)
    mlflow.log_param(
        "enabled_types",
        ", ".join(t.value for t in enabled),
    )
    mlflow.log_param(
        "input_path", config.input_documents_path
    )
    for qtype in enabled:
        qcfg = config.question_types[qtype]
        mlflow.log_param(
            f"count_{qtype.value}", qcfg.count
        )
        mlflow.log_param(
            f"difficulty_{qtype.value}", qcfg.difficulty
        )

    try:
        start_time = time.time()
        pipeline = Pipeline(config)
        output_file, test_set = pipeline.run()
        duration = time.time() - start_time

        # --- Core metrics ---
        questions = test_set.get("questions", [])
        actual_count = len(questions)
        mlflow.log_metric(
            "total_duration_seconds", round(duration, 1)
        )
        mlflow.log_metric(
            "questions_generated", actual_count
        )
        mlflow.log_metric(
            "questions_requested", total_q
        )
        mlflow.log_metric(
            "yield_ratio",
            round(actual_count / total_q, 3)
            if total_q > 0 else 0,
        )

        # --- Pipeline phase timings ---
        m = test_set.get(
            "metadata", {}
        ).get("metrics", {})
        mlflow.log_metric(
            "parse_time", round(m.get("parse_time", 0), 3)
        )
        mlflow.log_metric(
            "index_time", round(m.get("index_time", 0), 3)
        )
        mlflow.log_metric(
            "llm_calls", m.get("llm_calls", 0)
        )

        # --- Per-type counts ---
        by_type = m.get("questions_by_type", {})
        for qtype_name, cnt in by_type.items():
            mlflow.log_metric(
                f"count_{qtype_name}", cnt
            )

        # --- Quality metrics ---
        match_ratios = [
            q.get("metadata", {}).get(
                "context_match_ratio", 0
            )
            for q in questions
        ]
        if match_ratios:
            mlflow.log_metric(
                "avg_context_match_ratio",
                round(
                    sum(match_ratios) / len(match_ratios),
                    3,
                ),
            )
            mlflow.log_metric(
                "min_context_match_ratio",
                round(min(match_ratios), 3),
            )

        grounded = sum(
            1 for q in questions
            if q.get("metadata", {}).get(
                "answer_grounded", False
            )
        )
        if actual_count > 0:
            mlflow.log_metric(
                "answer_grounded_ratio",
                round(grounded / actual_count, 3),
            )

        # --- Artifact ---
        mlflow.log_artifact(output_file)

        mlflow.end_run()
        print(f"\nDone! Output: {output_file}")
        print("MLflow run completed")
        return output_file, test_set

    except Exception as e:
        print(f"ERROR: {e}")
        mlflow.log_param("error", str(e)[:250])
        mlflow.end_run(status="FAILED")
        raise


# ====================================================
# DEFAULT QUESTION TYPE PRESETS
# ====================================================
# Adjust counts and difficulties here, then re-run.

ALL_TYPES_DEFAULT = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="easy",
    ),
    QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
        enabled=True, count=3, difficulty="medium",
    ),
    QuestionType.SPECIFIC_JARGON: QuestionConfig(
        enabled=True, count=3, difficulty="medium",
    ),
    QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
        enabled=True, count=2, difficulty="hard",
    ),
    QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: QuestionConfig(
        enabled=True, count=2, difficulty="hard",
    ),
    QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
        enabled=False, count=2, difficulty="medium",
    ),
    QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
    QuestionType.LISTS_EXTRACTION: QuestionConfig(
        enabled=False, count=2, difficulty="easy",
    ),
    QuestionType.HALLUCINATION_TEST: QuestionConfig(
        enabled=False, count=2, difficulty="medium",
    ),
    QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
    QuestionType.PROMPT_INJECTION: QuestionConfig(
        enabled=False, count=2, difficulty="hard",
    ),
}

# ====================================================
# QUICK-RUN PRESETS — pick one or make your own
# ====================================================

QUICK_TEST = {
    QuestionType.DIRECT_LOOKUP: QuestionConfig(
        enabled=True, count=1, difficulty="easy",
    ),
    QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
        enabled=True, count=1, difficulty="medium",
    ),
}

FULL_RUN = {
    qt: QuestionConfig(enabled=True, count=5, difficulty=cfg.difficulty)
    for qt, cfg in ALL_TYPES_DEFAULT.items()
}


if __name__ == "__main__":

    # ================================================
    # CONFIGURATION - Adjust parameters here
    # ================================================

    # -- LLM model (used ONLY for Q+A generation) --
    MODEL = "gpt-5"

    # -- Naming --
    CORPUS_NAME = "DSL_corpus"

    # -- Question types to generate --
    # Set enabled=False or remove a type to disable it.
    QUESTION_TYPES = {
        QuestionType.DIRECT_LOOKUP: QuestionConfig(
            enabled=True, count=3, difficulty="easy"
        ),
        QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
            enabled=True, count=3, difficulty="medium"
        ),
        QuestionType.SPECIFIC_JARGON: QuestionConfig(
            enabled=True, count=3, difficulty="medium"
        ),
        QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
            enabled=True, count=3, difficulty="hard"
        ),
        QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: QuestionConfig(
            enabled=True, count=3, difficulty="hard"
        ),
        # QuestionType.CROSS_DOCUMENT_CONFLICT: QuestionConfig(
        #     enabled=True, count=2, difficulty="hard"
        # ),
        # QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
        #     enabled=True, count=2, difficulty="medium"
        # ),
        # QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
        #     enabled=True, count=2, difficulty="hard"
        # ),
        # QuestionType.LISTS_EXTRACTION: QuestionConfig(
        #     enabled=True, count=2, difficulty="easy"
        # ),
        # QuestionType.HALLUCINATION_TEST: QuestionConfig(
        #     enabled=True, count=2, difficulty="medium"
        # ),
        # QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
        #     enabled=True, count=2, difficulty="hard"
        # ),
        # QuestionType.PROMPT_INJECTION: QuestionConfig(
        #     enabled=True, count=2, difficulty="hard"
        # ),
    }

    # ================================================

    config = build_config(
        model=MODEL,
        corpus_name=CORPUS_NAME,
        question_types=QUESTION_TYPES,
        random_seed=42,
    )
    run_pipeline(config)
