"""
Main entry point for RLM Test Set Creator
Uses internal configuration - no argparse
"""
import os
from dotenv import load_dotenv

from src.config import TestSetConfig, QuestionType, QuestionConfig, RLMConfig
from src.test_set_creator import TestSetCreator

# Load environment variables
load_dotenv()

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

def main():
    """Main function with internal configuration"""
    
    # Define question types once - single source of truth
    enabled_question_types = {
        # ONLY first 4 question types enabled
        QuestionType.DIRECT_LOOKUP: QuestionConfig(
            enabled=True, count=10, difficulty="easy"
        ),
        QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(
            enabled=True, count=8, difficulty="medium"
        ),
        QuestionType.SPECIFIC_JARGON: QuestionConfig(
            enabled=True, count=7, difficulty="medium"
        ),
        QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(
            enabled=True, count=6, difficulty="hard"
        ),
        
        # ALL OTHER QUESTION TYPES DISABLED
        # QuestionType.CROSS_DOCUMENT_CONFLICT: QuestionConfig(
        #     enabled=True, count=5, difficulty="hard"
        # ),
        # QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(
        #     enabled=True, count=5, difficulty="medium"
        # ),
        # QuestionType.PINPOINTING_QUOTING: QuestionConfig(
        #     enabled=True, count=8, difficulty="medium"
        # ),
        # QuestionType.LONG_CONTEXT_SYNTHESIS: QuestionConfig(
        #     enabled=True, count=4, difficulty="hard"
        # ),
        # QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(
        #     enabled=True, count=6, difficulty="hard"
        # ),
        # QuestionType.AMBIGUOUS_QUESTIONS: QuestionConfig(
        #     enabled=True, count=5, difficulty="medium"
        # ),
        # QuestionType.HALLUCINATION_TEST: QuestionConfig(
        #     enabled=True, count=5, difficulty="medium"
        # ),
        # QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(
        #     enabled=False, count=4, difficulty="hard"
        # ),
        # QuestionType.PROMPT_INJECTION: QuestionConfig(
        #     enabled=False, count=3, difficulty="hard"
        # ),
        # QuestionType.TOOL_CALL_CHECK: QuestionConfig(
        #     enabled=True, count=4, difficulty="medium"
        # ),
        # QuestionType.TABLES_EXTRACTION: QuestionConfig(
        #     enabled=True, count=6, difficulty="medium"
        # ),
        # QuestionType.LISTS_EXTRACTION: QuestionConfig(
        #     enabled=True, count=6, difficulty="easy"
        # ),
        # QuestionType.INFOGRAPHIC_EXTRACTION: QuestionConfig(
        #     enabled=False, count=3, difficulty="hard"
        # ),
        # QuestionType.MULTI_TURN_FOLLOWUP: QuestionConfig(
        #     enabled=True, count=5, difficulty="medium"
        # ),
        # QuestionType.ACCESS_CONTROL: QuestionConfig(
        #     enabled=True, count=4, difficulty="medium"
        # ),
    }
    
    # Internal configuration - modify here to change settings
    config = TestSetConfig(
        # RLM settings
        rlm=RLMConfig(
            api_key=azure_key,
            model="gpt-5",
            recursive_model="gpt-5",
            max_iterations=5,  # Reduced from default (probably 20)
            enable_logging=False  # Disable verbose logging
        ),
        
        # File paths
        input_documents_path="data/files_for_test_set",
        output_path="data/test_sets",
        
        # Question type selection - single source of truth
        question_types=enabled_question_types,
        
        # General settings - use built-in method for consistency
        total_questions=0,  # Will be set by built-in method
        random_seed=42,
        
        # Metrics collection
        collect_latency=True,
        collect_accuracy=True
    )
    
    print("RLM Test Set Creator")
    print("=" * 50)
    print(f"Enabled question types: {len(config.get_enabled_question_types())}")
    print(f"Total questions to generate: {config.get_total_enabled_questions()}")
    print(f"Input path: {config.input_documents_path}")
    print(f"Output path: {config.output_path}")
    print(f"RLM model: {config.rlm.model}")
    print("=" * 50)
    
    # Update total_questions to match built-in calculation for consistency
    config.total_questions = config.get_total_enabled_questions()
    
    # Create and run test set creator
    creator = TestSetCreator(config)
    output_file = creator.run()
    
    print(f"\nTest set creation completed!")
    print(f"Output file: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()
