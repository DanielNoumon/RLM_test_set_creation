# RLM Test Set Creator

An automated test set creation system using Recursive Language Models (RLM) to generate comprehensive test questions for RAG applications.

## Features

### Question Types Supported
- **Direct Lookup** - Simple factual recall
- **Paraphrase Lookup** - Understanding rephrased questions  
- **Specific Jargon** - Product names, technical terms
- **Multi-hop within Corpus** - Requires multiple chunks
- **Cross-document Conflict** - Handling conflicting information
- **Temporal Questions** - Time-based queries
- **Pinpointing/Quoting** - Citation and section references
- **Long Context Synthesis** - Complex reasoning over large text
- **Needle-in-Haystack** - Finding specific facts in large context
- **Ambiguous Questions** - Handling unclear queries
- **Tool Call Check** - Computational questions
- **Tables Extraction** - Data table parsing
- **Lists Extraction** - Structured list information
- **Infographic Extraction** - Visual data interpretation
- **Hallucination Test** - Unanswerable questions
- **Adversarial/Aggro Handling** - Robustness testing
- **Prompt Injection Handling** - Security testing
- **Multi-turn Followup** - Memory and conversation
- **Access Control/Permissions** - Authorization testing

### Metadata Collection
- Expected behavior per question
- Golden answers for evaluation
- Required chunks/documents per question
- Latency and accuracy metrics
- Difficulty levels and question types

## Installation

1. Create conda environment:
```bash
conda create -n rlm_test_set_creation python=3.11
conda activate rlm_test_set_creation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

The main configuration is in `main.py` - modify the internal config section to enable/disable question types and adjust settings:

```python
# Enable/disable question types in the config
QuestionType.DIRECT_LOOKUP: QuestionConfig(
    enabled=True, count=10, difficulty="easy"
),
QuestionType.HALLUCINATION_TEST: QuestionConfig(
    enabled=True, count=5, difficulty="medium"
),
```

Run the test set creator:
```bash
python main.py
```

### Configuration Options

- **RLM Settings**: Model selection, iteration limits, logging
- **Question Types**: Enable/disable specific types, set counts and difficulty
- **File Paths**: Input documents and output locations
- **Metrics**: Latency and accuracy collection

### Input Documents

Place your documents in `data/documents/`:
- `.txt` files - plain text documents
- `.json` files - structured documents
- `.md` files - markdown documents

### Output

Test sets are saved to `data/test_sets/` as JSON files with:
- Complete question set with metadata
- Generation metrics and timing
- Configuration details for reproducibility

## Architecture

- `config.py` - Configuration management and question type definitions
- `question_generators.py` - Question generation logic for each type
- `test_set_creator.py` - Main pipeline and RLM integration
- `main.py` - Entry point with internal configuration

## RLM Integration

The system is designed to integrate with the rlm-minimal architecture:
- Uses RLM_REPL for sophisticated question generation
- Recursive reasoning for complex question types
- Tool execution for computational questions

## Example Output

```json
{
  "metadata": {
    "created_at": 1640995200,
    "config": {
      "rlm_model": "gpt-4",
      "enabled_types": ["direct_lookup", "multi_hop_within_corpus"],
      "total_questions": 50
    },
    "metrics": {
      "generation_time": 12.34,
      "questions_by_type": {
        "direct_lookup": 10,
        "multi_hop_within_corpus": 6
      }
    }
  },
  "questions": [
    {
      "id": "direct_lookup_0",
      "type": "direct_lookup",
      "question": "What information is mentioned about: Project Phoenix started...",
      "expected_behavior": "Direct factual recall from text",
      "golden_answer": "Project Phoenix started in 2023 with a budget...",
      "relevant_chunks": [0],
      "difficulty": "easy",
      "metadata": {
        "source_chunk": 0,
        "question_type": "direct_lookup"
      }
    }
  ]
}
```

## Customization

### Adding New Question Types

1. Add to `QuestionType` enum in `config.py`
2. Create generator class in `question_generators.py`
3. Register in `QuestionGeneratorFactory`
4. Enable in main configuration

### Custom Generators

Implement the `QuestionGenerator` base class:

```python
class CustomQuestionGenerator(QuestionGenerator):
    def generate_questions(self, documents, count):
        # Your custom logic here
        return questions
```

## License

MIT License - see LICENSE file for details.
