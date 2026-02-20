"""
RLM integration using actual rlm-minimal repository
"""
import sys
import os
from typing import List, Dict, Any

# Add rlm-minimal to path if it exists
# Go up one level from src/ to find rlm-minimal-main/
rlm_path = os.path.join(os.path.dirname(__file__), '..', 'rlm-minimal-main')
rlm_path = os.path.abspath(rlm_path)
if os.path.exists(rlm_path):
    sys.path.insert(0, rlm_path)
    print(f"✓ Added rlm-minimal path: {rlm_path}")
else:
    print(f"✗ rlm-minimal path not found: {rlm_path}")

# Try importing with the path added
try:
    from rlm.rlm_repl import RLM_REPL
    RLM_AVAILABLE = True
    print("✓ RLM_REPL imported successfully")
except ImportError as e:
    RLM_AVAILABLE = False
    print(f"✗ Failed to import RLM_REPL: {e}")
    # Try importing without path (maybe it's already in Python path)
    try:
        import sys
        print("Trying import without explicit path...")
        from rlm.rlm_repl import RLM_REPL
        RLM_AVAILABLE = True
        print("✓ RLM_REPL imported successfully (without explicit path)")
    except ImportError as e2:
        print(f"✗ Still failed: {e2}")

from .config import TestSetConfig, QuestionType

class RLMIntegration:
    """Integration with actual RLM_REPL from rlm-minimal"""
    
    def __init__(self, config: TestSetConfig):
        self.config = config.rlm
        
        # Validate RLM availability
        if not RLM_AVAILABLE:
            raise ImportError(
                "rlm-minimal repository not found. Please clone rlm-minimal to use real RLM functionality. "
                "Expected location: rlm-minimal-main folder or install as package"
            )
        
        if not config.rlm.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY environment variable to use RLM"
            )
        
        # Initialize real RLM
        try:
            self.rlm = RLM_REPL(
                api_key=config.rlm.api_key,
                model=config.rlm.model,
                recursive_model=config.rlm.recursive_model,
                max_iterations=config.rlm.max_iterations,
                enable_logging=config.rlm.enable_logging
            )
            self.use_real_rlm = True
            print("✓ RLM_REPL successfully initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RLM_REPL: {e}")
    
    def generate_questions_with_rlm(self, documents: List[Dict[str, Any]], 
                                  question_type: QuestionType, 
                                  count: int) -> List[Dict[str, Any]]:
        """Generate questions using RLM"""
        if not self.use_real_rlm:
            raise RuntimeError("RLM not properly initialized")
        
        return self._generate_with_real_rlm(documents, question_type, count)
    
    def _generate_with_real_rlm(self, documents: List[Dict[str, Any]], 
                               question_type: QuestionType, 
                               count: int) -> List[Dict[str, Any]]:
        """Use actual RLM_REPL to generate questions"""
        questions = []
        
        # Prepare context for RLM
        context_text = self._prepare_context(documents)
        
        # Create prompt for question generation
        prompt = self._create_generation_prompt(question_type, count)
        
        # Add timeout and progress tracking
        import time
        start_time = time.time()
        timeout = 300  # 5 minutes max per question
        
        for i in range(count):
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"Timeout reached after {timeout} seconds")
                break
                
            try:
                print(f"Generating question {i+1}/{count}...")
                
                # Use RLM to generate a question
                response = self.rlm.completion(context=context_text, query=prompt)
                
                # Parse the response to extract question details
                question_data = self._parse_rlm_response(response, question_type, i, documents)
                if question_data:
                    questions.append(question_data)
                    print(f"✓ Generated question {i+1}")
                else:
                    print(f"Warning: Failed to parse RLM response for question {i}")
                    
            except Exception as e:
                error_msg = f"Error generating question {i} with RLM: {e}"
                print(f"ERROR: {error_msg}")
                # Log the error but don't create fallback - let the error propagate
                if i == 0:  # If first question fails, likely a systemic issue
                    raise RuntimeError(f"RLM question generation failed: {error_msg}")
                continue
        
        if not questions:
            raise RuntimeError("No questions could be generated with RLM")
        
        return questions
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare documents as context for RLM"""
        context_parts = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                content = doc
            elif isinstance(doc, dict):
                content = doc.get('content', str(doc))
            
            # Clean up Unicode characters
            clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
            context_parts.append(f"Document {i+1}:\n{clean_content}")
        
        return "\n\n".join(context_parts)
    
    def _create_generation_prompt(self, question_type: QuestionType, count: int) -> str:
        """Create prompt for generating specific question type"""
        prompts = {
            QuestionType.DIRECT_LOOKUP: (
                "Generate a direct lookup question that can be answered by finding "
                "specific information in the provided documents. The question should "
                "be straightforward and require factual recall."
            ),
            QuestionType.PARAPHRASE_LOOKUP: (
                "Generate a question that uses different wording than the text but "
                "requires finding the same information. The question should test "
                "understanding of paraphrased queries."
            ),
            QuestionType.MULTI_HOP_WITHIN_CORPUS: (
                "Generate a question that requires combining information from multiple "
                "documents or sections. The answer should synthesize information from "
                "different sources."
            ),
            QuestionType.NEEDLE_IN_HAYSTACK: (
                "Generate a question about a very specific detail that would be hard "
                "to find in a large context. The question should test precise "
                "information retrieval."
            ),
            QuestionType.HALLUCINATION_TEST: (
                "Generate a question that cannot be answered from the provided documents. "
                "The question should be about something not mentioned in the context."
            )
        }
        
        base_prompt = prompts.get(question_type, f"Generate a {question_type.value} question.")
        return f"{base_prompt}\n\nPlease provide:\n1. The question\n2. The expected answer\n3. Which documents contain the answer"
    
    def _parse_rlm_response(self, response: str, question_type: QuestionType, 
                           index: int, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse RLM response into structured question format"""
        # This is a simplified parser - in practice you'd want more sophisticated parsing
        lines = response.strip().split('\n')
        
        question = ""
        answer = ""
        relevant_docs = []
        
        # Try to extract structured information
        for line in lines:
            if line.lower().startswith('question:'):
                question = line.replace('question:', '').strip()
            elif line.lower().startswith('answer:'):
                answer = line.replace('answer:', '').strip()
            elif line.lower().startswith('documents:'):
                docs_str = line.replace('documents:', '').strip()
                try:
                    relevant_docs = [int(d.strip()) for d in docs_str.split(',')]
                except:
                    relevant_docs = []
        
        # Fallback if structured parsing failed
        if not question:
            question = response.strip().split('\n')[0] if response else f"Generated question {index}"
        if not answer:
            answer = response.strip() if response else f"Generated answer {index}"
        if not relevant_docs:
            relevant_docs = [index % len(documents)] if documents else []
        
        return {
            "id": f"rlm_{question_type.value}_{index}",
            "type": question_type.value,
            "question": question,
            "expected_behavior": f"RLM-generated {question_type.value}",
            "golden_answer": answer,
            "relevant_chunks": relevant_docs,
            "difficulty": "medium",
            "metadata": {
                "generated_by": "RLM_REPL",
                "question_type": question_type.value,
                "rlm_model": self.config.model,
                "recursive_model": self.config.recursive_model
            }
        }
    
    def _create_fallback_question(self, question_type: QuestionType, index: int, 
                                 documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback question when RLM fails"""
        return {
            "id": f"fallback_{question_type.value}_{index}",
            "type": question_type.value,
            "question": f"Fallback {question_type.value} question {index}",
            "expected_behavior": f"Fallback {question_type.value}",
            "golden_answer": f"Fallback answer for {question_type.value} {index}",
            "relevant_chunks": [index % len(documents)] if documents else [],
            "difficulty": "medium",
            "metadata": {
                "generated_by": "Fallback",
                "question_type": question_type.value,
                "error": "RLM generation failed"
            }
        }
