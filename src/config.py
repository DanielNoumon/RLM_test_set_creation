"""
Configuration for RLM Test Set Creator
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class QuestionType(Enum):
    DIRECT_LOOKUP = "direct_lookup"
    PARAPHRASE_LOOKUP = "paraphrase_lookup"
    SPECIFIC_JARGON = "specific_jargon"
    MULTI_HOP_WITHIN_CORPUS = "multi_hop_within_corpus"
    CROSS_DOCUMENT_CONFLICT = "cross_document_conflict"
    TEMPORAL_QUESTIONS = "temporal_questions"
    PINPOINTING_QUOTING = "pinpointing_quoting"
    LONG_CONTEXT_SYNTHESIS = "long_context_synthesis"
    NEEDLE_IN_HAYSTACK = "needle_in_haystack"
    AMBIGUOUS_QUESTIONS = "ambiguous_questions"
    TOOL_CALL_CHECK = "tool_call_check"
    TABLES_EXTRACTION = "tables_extraction"
    LISTS_EXTRACTION = "lists_extraction"
    INFOGRAPHIC_EXTRACTION = "infographic_extraction"
    HALLUCINATION_TEST = "hallucination_test"
    ADVERSARIAL_AGGRO = "adversarial_aggro"
    PROMPT_INJECTION = "prompt_injection"
    MULTI_TURN_FOLLOWUP = "multi_turn_followup"
    ACCESS_CONTROL = "access_control"

@dataclass
class QuestionConfig:
    """Configuration for individual question types"""
    enabled: bool = False
    count: int = 5
    difficulty: str = "medium"  # easy, medium, hard
    expected_behavior: Optional[str] = None
    metadata_fields: List[str] = field(default_factory=list)

@dataclass
class RLMConfig:
    """Configuration for RLM model"""
    model: str = "gpt-4"
    recursive_model: str = "gpt-3.5-turbo"
    max_iterations: int = 10
    enable_logging: bool = True
    api_key: Optional[str] = None

@dataclass
class TestSetConfig:
    """Main configuration for test set creation"""
    # RLM settings
    rlm: RLMConfig = field(default_factory=RLMConfig)
    
    # Input settings
    input_documents_path: str = "data/documents"
    output_path: str = "data/test_sets"
    
    # Question type configurations
    question_types: Dict[QuestionType, QuestionConfig] = field(default_factory=dict)
    
    # General settings
    total_questions: int = 100
    random_seed: int = 42
    
    # Metrics collection
    collect_latency: bool = True
    collect_accuracy: bool = True
    
    def __post_init__(self):
        """Initialize default question type configurations"""
        if not self.question_types:
            # Enable all question types by default with reasonable counts
            default_configs = {
                QuestionType.DIRECT_LOOKUP: QuestionConfig(enabled=True, count=10),
                QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(enabled=True, count=8),
                QuestionType.SPECIFIC_JARGON: QuestionConfig(enabled=True, count=7),
                QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(enabled=True, count=6),
                QuestionType.CROSS_DOCUMENT_CONFLICT: QuestionConfig(enabled=True, count=5),
                QuestionType.TEMPORAL_QUESTIONS: QuestionConfig(enabled=True, count=5),
                QuestionType.PINPOINTING_QUOTING: QuestionConfig(enabled=True, count=8),
                QuestionType.LONG_CONTEXT_SYNTHESIS: QuestionConfig(enabled=True, count=4),
                QuestionType.NEEDLE_IN_HAYSTACK: QuestionConfig(enabled=True, count=6),
                QuestionType.AMBIGUOUS_QUESTIONS: QuestionConfig(enabled=True, count=5),
                QuestionType.TOOL_CALL_CHECK: QuestionConfig(enabled=True, count=4),
                QuestionType.TABLES_EXTRACTION: QuestionConfig(enabled=True, count=6),
                QuestionType.LISTS_EXTRACTION: QuestionConfig(enabled=True, count=6),
                QuestionType.INFOGRAPHIC_EXTRACTION: QuestionConfig(enabled=True, count=3),
                QuestionType.HALLUCINATION_TEST: QuestionConfig(enabled=True, count=5),
                QuestionType.ADVERSARIAL_AGGRO: QuestionConfig(enabled=True, count=4),
                QuestionType.PROMPT_INJECTION: QuestionConfig(enabled=True, count=3),
                QuestionType.MULTI_TURN_FOLLOWUP: QuestionConfig(enabled=True, count=5),
                QuestionType.ACCESS_CONTROL: QuestionConfig(enabled=True, count=4),
            }
            self.question_types = default_configs
    
    def get_enabled_question_types(self) -> List[QuestionType]:
        """Get list of enabled question types"""
        return [qtype for qtype, config in self.question_types.items() if config.enabled]
    
    def get_total_enabled_questions(self) -> int:
        """Get total number of questions for enabled types"""
        return sum(config.count for config in self.question_types.values() if config.enabled)

# Default configuration instance
DEFAULT_CONFIG = TestSetConfig()

# Example custom configurations
SIMPLE_CONFIG = TestSetConfig(
    question_types={
        QuestionType.DIRECT_LOOKUP: QuestionConfig(enabled=True, count=20),
        QuestionType.PARAPHRASE_LOOKUP: QuestionConfig(enabled=True, count=15),
        QuestionType.MULTI_HOP_WITHIN_CORPUS: QuestionConfig(enabled=True, count=10),
    }
)

COMPREHENSIVE_CONFIG = TestSetConfig(
    total_questions=200,
    question_types={
        qtype: QuestionConfig(enabled=True, count=10) 
        for qtype in QuestionType
    }
)
