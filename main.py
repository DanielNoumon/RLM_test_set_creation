"""
Main entry point for RLM Test Set Creator
Uses internal configuration - no argparse
"""
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import mlflow

from src.config import TestSetConfig, QuestionType, QuestionConfig, RLMConfig
from src.test_set_creator import TestSetCreator

# Load environment variables
load_dotenv()

# Configure MLflow tracking
mlflow.openai.autolog()
mlflow_db = Path(__file__).parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.set_experiment("rlm-test-set-creation")
mlflow.disable_system_metrics_logging()
print(f"MLflow tracking enabled (SQLite): {mlflow_db}")

# Set OPENAI_API_KEY for rlm-minimal compatibility
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
if azure_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = azure_key

# Set Azure OpenAI environment variables for rlm-minimal
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_GPT5")
if azure_endpoint and not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION_GPT5")
if azure_api_version and not os.getenv("AZURE_OPENAI_API_VERSION"):
    os.environ["AZURE_OPENAI_API_VERSION"] = azure_api_version

def main(config: TestSetConfig):
    """Run the test set creation pipeline."""
    print("RLM Test Set Creator")
    print("=" * 50)
    enabled = config.get_enabled_question_types()
    total_q = config.get_total_enabled_questions()
    print(f"Enabled question types: {len(enabled)}")
    print(f"Total questions to generate: {total_q}")
    print(f"Input path: {config.input_documents_path}")
    print(f"Output path: {config.output_path}")
    print(f"RLM model: {config.rlm.model}")
    print(f"Recursive model: {config.rlm.recursive_model}")
    print(f"Max iterations: {config.rlm.max_iterations}")
    print("=" * 50)

    config.total_questions = total_q

    # Start MLflow run with human-readable name
    run_name = (
        f"{config.corpus_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    mlflow.start_run(run_name=run_name)

    # Log configuration params
    mlflow.log_param("model", config.rlm.model)
    mlflow.log_param(
        "recursive_model", config.rlm.recursive_model
    )
    mlflow.log_param(
        "max_iterations", config.rlm.max_iterations
    )
    mlflow.log_param(
        "corpus_name", config.corpus_name
    )
    mlflow.log_param(
        "questions_requested", total_q
    )
    mlflow.log_param(
        "enabled_types",
        ", ".join(t.value for t in enabled)
    )
    mlflow.log_param(
        "input_path", config.input_documents_path
    )
    mlflow.log_param("pipeline_version", "3-phase")
    # Log per-type difficulty settings
    for qtype in enabled:
        qcfg = config.question_types[qtype]
        mlflow.log_param(
            f"difficulty_{qtype.value}", qcfg.difficulty
        )

    try:
        start_time = time.time()
        creator = TestSetCreator(config)
        output_file, test_set = creator.run()
        duration = time.time() - start_time

        # --- Core metrics ---
        mlflow.log_metric(
            "total_duration_seconds", duration
        )
        questions = test_set.get("questions", [])
        actual_count = len(questions)
        mlflow.log_metric(
            "questions_generated", actual_count
        )

        # --- Per-type counts ---
        by_type = test_set.get(
            "metadata", {}
        ).get("metrics", {}).get("questions_by_type", {})
        for qtype_name, cnt in by_type.items():
            mlflow.log_metric(
                f"count_{qtype_name}", cnt
            )

        # --- Quality metrics (3-phase pipeline) ---
        # Context match ratio: how well passages match docs
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
                    3
                )
            )

        # Answer grounding: how many answers are grounded
        grounded = sum(
            1 for q in questions
            if q.get("metadata", {}).get(
                "answer_grounded", False
            )
        )
        if actual_count > 0:
            mlflow.log_metric(
                "answer_grounded_count", grounded
            )
            mlflow.log_metric(
                "answer_grounded_ratio",
                round(grounded / actual_count, 3)
            )

        # Backward-compatible: hallucination/repair counts
        # (should be 0 with 3-phase pipeline)
        hallucinations = sum(
            1 for q in questions
            if q.get("hallucination_detected", False)
        )
        mlflow.log_metric(
            "hallucinations_detected", hallucinations
        )
        if actual_count > 0:
            mlflow.log_metric(
                "clean_ratio",
                round(
                    (actual_count - hallucinations)
                    / actual_count, 3
                )
            )

        # --- Artifact ---
        mlflow.log_artifact(output_file)

        print(f"\nTest set creation completed in {duration:.1f}s")
        print(f"Output file: {output_file}")

        mlflow.end_run()
        print("MLflow run completed")
        return output_file

    except Exception as e:
        print(f"ERROR: {e}")
        mlflow.log_param("error", str(e)[:250])
        mlflow.end_run(status="FAILED")
        raise


if __name__ == "__main__":

    # ================================================
    # CONFIGURATION - Adjust parameters here
    # ================================================

    # -- LLM models --
    MODEL = "gpt-5"                # Root LLM for orchestration
    RECURSIVE_MODEL = "gpt-5" # Sub-LLM for document extraction
    MAX_ITERATIONS = 5             # Max REPL iterations per RLM call - if you scale up the input documents, you'd likely want to increase MAX_ITERATIONS (e.g., 8-10) to give the RLM enough room to explore.
    ENABLE_LOGGING = True          # Show RLM agent reasoning

    # -- Naming --
    CORPUS_NAME = "DSL_corpus"     # Name of the document corpus (used in output filename)

    # -- Paths --
    INPUT_DOCUMENTS_PATH = "data/files_for_test_set"
    OUTPUT_PATH = "data/test_sets"

    # -- Question types to generate --
    # Set count=0 or remove a type to disable it.
    # Each type makes ceil(count / 5) RLM calls (batch size = 5).
    QUESTION_TYPES = {
        QuestionType.DIRECT_LOOKUP: QuestionConfig(
            enabled=True, count=1, difficulty="easy"
        ),
        QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
            enabled=True, count=1, difficulty="medium"
        ),
        QuestionType.SPECIFIC_JARGON: QuestionConfig(
            enabled=True, count=1, difficulty="medium"
        ),
        QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
            enabled=True, count=1, difficulty="hard"
        ),
        QuestionType.MULTI_HOP_BETWEEN_DOCUMENTS: QuestionConfig(
            enabled=True, count=1, difficulty="hard"
        ),
        # Uncomment to enable more types:
        # QuestionType.CROSS_DOCUMENT_CONFLICT: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.PINPOINTING_QUOTING: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.LONG_CONTEXT_SYNTHESIS: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.AMBIGUOUS_QUESTIONS: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.HALLUCINATION_TEST: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.PROMPT_INJECTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.TOOL_CALL_CHECK: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.TABLES_EXTRACTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.LISTS_EXTRACTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="easy"
        # ),
        # QuestionType.INFOGRAPHIC_EXTRACTION: QuestionConfig(
        #     enabled=True, count=3, difficulty="hard"
        # ),
        # QuestionType.MULTI_TURN_FOLLOWUP: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
        # QuestionType.ACCESS_CONTROL: QuestionConfig(
        #     enabled=True, count=3, difficulty="medium"
        # ),
    }

    # ================================================

    config = TestSetConfig(
        rlm=RLMConfig(
            api_key=azure_key,
            model=MODEL,
            recursive_model=RECURSIVE_MODEL,
            max_iterations=MAX_ITERATIONS,
            enable_logging=ENABLE_LOGGING,
        ),
        corpus_name=CORPUS_NAME,
        input_documents_path=INPUT_DOCUMENTS_PATH,
        output_path=OUTPUT_PATH,
        question_types=QUESTION_TYPES,
        random_seed=42,
    )

    main(config)
