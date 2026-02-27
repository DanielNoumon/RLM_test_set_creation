"""
Question generators for different test question types
"""
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import random
import json
from datetime import datetime

from .config import QuestionType, QuestionConfig

class QuestionGenerator(ABC):
    """Base class for question generators"""
    
    def __init__(self, config: QuestionConfig):
        self.config = config
        self.random = random.Random(42)  # For reproducibility
    
    @abstractmethod
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """Generate questions of this type"""
        pass
    
    def _extract_text_chunks(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract text chunks from documents"""
        chunks = []
        for doc in documents:
            if isinstance(doc, str):
                chunks.append(doc)
            elif isinstance(doc, dict) and 'content' in doc:
                chunks.append(doc['content'])
            elif isinstance(doc, dict) and 'text' in doc:
                chunks.append(doc['text'])
        return chunks
    
    def _find_relevant_chunks(self, question: str, chunks: List[str]) -> List[int]:
        """Find chunks most relevant to a question (simplified)"""
        # In a real implementation, this would use embeddings or semantic search
        # For now, return random chunks as placeholder
        return self.random.sample(range(len(chunks)), min(3, len(chunks)))

class DirectLookupGenerator(QuestionGenerator):
    """Generate direct lookup questions"""
    
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        chunks = self._extract_text_chunks(documents)
        questions = []
        
        for i in range(min(count, len(chunks))):
            chunk = chunks[i]
            # Extract a key fact from the chunk (simplified)
            words = chunk.split()
            if len(words) > 10:
                # Take a random sentence or phrase
                start = self.random.randint(0, max(0, len(words) - 20))
                end = min(start + 15, len(words))
                fact = " ".join(words[start:end])
                
                question = {
                    "id": f"direct_lookup_{i}",
                    "type": QuestionType.DIRECT_LOOKUP.value,
                    "question": f"What information is mentioned about: {fact[:50]}...?",
                    "expected_behavior": "Direct factual recall from text",
                    "golden_answer": fact,
                    "relevant_chunks": [i],
                    "difficulty": self.config.difficulty,
                    "metadata": {
                        "source_chunk": i,
                        "fact_length": len(fact),
                        "question_type": "direct_lookup"
                    }
                }
                questions.append(question)
        
        return questions

class ParaphraseLookupGenerator(QuestionGenerator):
    """Generate paraphrase lookup questions"""
    
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        chunks = self._extract_text_chunks(documents)
        questions = []
        
        for i in range(min(count, len(chunks))):
            chunk = chunks[i]
            words = chunk.split()
            if len(words) > 10:
                # Extract a fact and create paraphrased question
                start = self.random.randint(0, max(0, len(words) - 20))
                end = min(start + 15, len(words))
                fact = " ".join(words[start:end])
                
                # Create paraphrased question templates
                templates = [
                    f"Can you explain what is meant by: {fact[:30]}...?",
                    f"What details are provided regarding: {fact[:30]}...?",
                    f"Describe the information about: {fact[:30]}...",
                    f"What can you tell me about: {fact[:30]}...?"
                ]
                
                question = {
                    "id": f"paraphrase_lookup_{i}",
                    "type": QuestionType.PARAPHRASE_LOOKUP.value,
                    "question": self.random.choice(templates),
                    "expected_behavior": "Understanding paraphrased questions",
                    "golden_answer": fact,
                    "relevant_chunks": [i],
                    "difficulty": self.config.difficulty,
                    "metadata": {
                        "source_chunk": i,
                        "paraphrase_type": "rephrased",
                        "question_type": "paraphrase_lookup"
                    }
                }
                questions.append(question)
        
        return questions

class MultiHopGenerator(QuestionGenerator):
    """Generate multi-hop questions requiring information from multiple chunks"""
    
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        chunks = self._extract_text_chunks(documents)
        questions = []
        
        for i in range(min(count, len(chunks) // 2)):
            # Select 2-3 related chunks
            num_chunks = min(3, len(chunks))
            selected_chunks = self.random.sample(range(len(chunks)), num_chunks)
            
            # Create a question that requires combining information
            chunk_facts = []
            for chunk_idx in selected_chunks:
                words = chunks[chunk_idx].split()
                if len(words) > 5:
                    start = self.random.randint(0, max(0, len(words) - 10))
                    end = min(start + 8, len(words))
                    chunk_facts.append(" ".join(words[start:end]))
            
            if len(chunk_facts) >= 2:
                question = {
                    "id": f"multi_hop_{i}",
                    "type": QuestionType.MULTI_HOP_WITHIN_CORPUS.value,
                    "question": f"How do the following facts relate: {chunk_facts[0][:30]}... and {chunk_facts[1][:30]}...?",
                    "expected_behavior": "Synthesize information from multiple sources",
                    "golden_answer": f"Combines information from: {'; '.join(chunk_facts)}",
                    "relevant_chunks": selected_chunks,
                    "difficulty": self.config.difficulty,
                    "metadata": {
                        "required_chunks": len(selected_chunks),
                        "synthesis_type": "multi_hop",
                        "question_type": "multi_hop_within_corpus"
                    }
                }
                questions.append(question)
        
        return questions

class NeedleInHaystackGenerator(QuestionGenerator):
    """Generate needle-in-haystack questions"""
    
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        chunks = self._extract_text_chunks(documents)
        questions = []
        
        # Create a "needle" - specific information to find
        needles = [
            "The magic number is 42",
            "Project Phoenix started in 2023",
            "The budget was exactly $1,234,567",
            "Meeting scheduled for March 15th",
            "The password is 'unicorn123'"
        ]
        
        for i in range(min(count, len(needles))):
            needle = needles[i % len(needles)]
            haystack_chunk = self.random.choice(chunks)
            
            # Insert needle into haystack (in real implementation, this would be more sophisticated)
            modified_chunk = haystack_chunk + " " + needle
            
            question = {
                "id": f"needle_haystack_{i}",
                "type": QuestionType.NEEDLE_IN_HAYSTACK.value,
                "question": f"Find the specific information about: {needle.split()[2]}",
                "expected_behavior": "Find specific fact in large context",
                "golden_answer": needle,
                "relevant_chunks": [chunks.index(haystack_chunk)],
                "difficulty": self.config.difficulty,
                "metadata": {
                    "needle": needle,
                    "haystack_size": len(modified_chunk),
                    "question_type": "needle_in_haystack"
                }
            }
            questions.append(question)
        
        return questions

class HallucinationTestGenerator(QuestionGenerator):
    """Generate questions that cannot be answered from the documents"""
    
    def generate_questions(self, documents: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        questions = []
        
        # Create questions about information not in documents
        unanswerable_topics = [
            "What is the weather on Mars?",
            "Who won the 2050 World Cup?",
            "What is the meaning of life according to these documents?",
            "How much did the company spend on coffee in 1850?",
            "What are the quantum computing specifications mentioned?"
        ]
        
        for i in range(min(count, len(unanswerable_topics))):
            topic = unanswerable_topics[i % len(unanswerable_topics)]
            
            question = {
                "id": f"hallucination_test_{i}",
                "type": QuestionType.HALLUCINATION_TEST.value,
                "question": topic,
                "expected_behavior": "Should indicate information not found",
                "golden_answer": "Information not found in documents",
                "relevant_chunks": [],
                "difficulty": self.config.difficulty,
                "metadata": {
                    "unanswerable": True,
                    "expected_response": "not_found",
                    "question_type": "hallucination_test"
                }
            }
            questions.append(question)
        
        return questions

class QuestionGeneratorFactory:
    """Factory to create question generators"""
    
    _generators = {
        QuestionType.DIRECT_LOOKUP: DirectLookupGenerator,
        QuestionType.PARAPHRASE_LOOKUP: ParaphraseLookupGenerator,
        QuestionType.MULTI_HOP_WITHIN_CORPUS: MultiHopGenerator,
        QuestionType.NEEDLE_IN_HAYSTACK: NeedleInHaystackGenerator,
        QuestionType.HALLUCINATION_TEST: HallucinationTestGenerator,
    }
    
    @classmethod
    def get_generator(cls, question_type: QuestionType, config: QuestionConfig) -> QuestionGenerator:
        """Get generator for question type"""
        if question_type not in cls._generators:
            raise ValueError(f"No generator available for {question_type}")
        
        return cls._generators[question_type](config)
    
    @classmethod
    def register_generator(cls, question_type: QuestionType, generator_class: type):
        """Register a new generator"""
        cls._generators[question_type] = generator_class
