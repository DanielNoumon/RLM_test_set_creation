"""
RLM-based Test Set Creator
"""
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from .config import TestSetConfig, QuestionType, DEFAULT_CONFIG
from .question_generators import QuestionGeneratorFactory

from .rlm_integration import RLMIntegration

class TestSetCreator:
    """Main test set creation pipeline"""
    
    def __init__(self, config: TestSetConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.rlm = RLMIntegration(self.config)
        self.start_time = None
        
    def load_documents(self, documents_path: str) -> List[Dict[str, Any]]:
        """Load documents from specified path"""
        documents = []
        path = Path(documents_path)
        
        if not path.exists():
            print(f"Creating sample documents at {documents_path}")
            return self._create_sample_documents(documents_path)
        
        # Load from various formats
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.json', '.md', '.pdf']:
                try:
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            if isinstance(content, list):
                                documents.extend(content)
                            else:
                                documents.append(content)
                    elif file_path.suffix == '.pdf':
                        # Handle PDF files - extract text using PyMuPDF
                        try:
                            import fitz  # PyMuPDF
                            with fitz.open(file_path) as doc:
                                text = ""
                                for page in doc:
                                    page_text = page.get_text()
                                    # Clean up Unicode characters
                                    text += page_text.encode('utf-8', errors='ignore').decode('utf-8') + "\n"
                                documents.append({
                                    'filename': str(file_path),
                                    'content': text,
                                    'type': 'pdf'
                                })
                        except ImportError:
                            print(f"PyMuPDF not installed. Skipping PDF: {file_path}")
                        except Exception as e:
                            print(f"Error reading PDF {file_path}: {e}")
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            documents.append({
                                'filename': str(file_path),
                                'content': f.read(),
                                'type': 'text'
                            })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _create_sample_documents(self, documents_path: str) -> List[Dict[str, Any]]:
        """Create sample documents for testing"""
        os.makedirs(documents_path, exist_ok=True)
        
        sample_docs = [
            {
                "title": "Project Overview",
                "content": "Project Phoenix started in 2023 with a budget of $1,234,567. The main goal is to develop advanced AI systems for automated testing. The team consists of 15 engineers and researchers.",
                "type": "overview"
            },
            {
                "title": "Technical Specifications",
                "content": "The system uses GPT-4 as the primary model with support for recursive reasoning. Memory requirements are 32GB RAM minimum. Processing speed is approximately 1000 tokens per second.",
                "type": "technical"
            },
            {
                "title": "Meeting Notes",
                "content": "Meeting scheduled for March 15th to discuss Q2 milestones. Key topics include budget allocation, timeline adjustments, and resource planning. The password for the development server is 'unicorn123'.",
                "type": "meeting"
            },
            {
                "title": "Budget Report",
                "content": "Total budget: $1,234,567. Q1 spending: $234,500. Remaining budget: $1,000,067. Major expenses include hardware purchases and software licenses.",
                "type": "financial"
            },
            {
                "title": "Team Structure",
                "content": "Engineering team: 8 members. Research team: 4 members. Management: 3 members. The magic number for the project success metric is 42.",
                "type": "organizational"
            }
        ]
        
        # Save sample documents
        for i, doc in enumerate(sample_docs):
            with open(f"{documents_path}/doc_{i}.json", 'w') as f:
                json.dump(doc, f, indent=2)
        
        return sample_docs
    
    def generate_test_set(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate complete test set"""
        self.start_time = time.time()
        all_questions = []
        metrics = {
            "generation_time": 0,
            "questions_by_type": {},
            "total_questions": 0
        }
        
        enabled_types = self.config.get_enabled_question_types()
        print(f"Generating questions for {len(enabled_types)} types")
        
        covered_topics: List[str] = []
        covered_documents: List[str] = []

        for question_type in enabled_types:
            config = self.config.question_types[question_type]
            print(f"Generating {config.count} {question_type.value} questions...")
            
            # Use RLM for sophisticated question generation
            if hasattr(self.rlm, 'generate_questions_with_rlm'):
                questions = self.rlm.generate_questions_with_rlm(
                    documents, question_type, config.count,
                    difficulty=config.difficulty,
                    covered_topics=covered_topics,
                    covered_documents=covered_documents
                )
            else:
                # Fallback to basic generators
                generator = QuestionGeneratorFactory.get_generator(question_type, config)
                questions = generator.generate_questions(documents, config.count)
            
            # Track topics and documents for diversity
            for q in questions:
                covered_topics.append(q["question"])
                for src in q.get("source_documents", []):
                    if src not in covered_documents:
                        covered_documents.append(src)

            all_questions.extend(questions)
            metrics["questions_by_type"][question_type.value] = len(questions)
        
        metrics["generation_time"] = time.time() - self.start_time
        metrics["total_questions"] = len(all_questions)
        
        test_set = {
            "metadata": {
                "created_at": time.time(),
                "config": {
                    "rlm_model": self.config.rlm.model,
                    "enabled_types": [t.value for t in enabled_types],
                    "total_questions": metrics["total_questions"]
                },
                "metrics": metrics
            },
            "questions": all_questions
        }
        
        return test_set
    
    def save_test_set(self, test_set: Dict[str, Any], output_path: str):
        """Save test set to file"""
        name = self.config.corpus_name.replace(" ", "_")
        corpus_dir = os.path.join(output_path, name)
        os.makedirs(corpus_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%d_%m_%y_T%H_%M")
        filename = f"{name}_{date_str}.json"
        filepath = os.path.join(corpus_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_set, f, indent=2, ensure_ascii=False)
        
        print(f"Test set saved to {filepath}")
        print(f"Generated {test_set['metadata']['metrics']['total_questions']} questions")
        print(f"Generation time: {test_set['metadata']['metrics']['generation_time']:.2f}s")
        
        return filepath
    
    def run(self):
        """Run the complete test set creation pipeline.

        Returns:
            Tuple of (output_file_path, test_set_dict).
        """
        print("Starting RLM Test Set Creator...")
        
        # Load documents
        documents = self.load_documents(self.config.input_documents_path)
        print(f"Loaded {len(documents)} documents")
        
        # Generate test set
        test_set = self.generate_test_set(documents)
        
        # Save results
        output_file = self.save_test_set(test_set, self.config.output_path)
        
        return output_file, test_set
