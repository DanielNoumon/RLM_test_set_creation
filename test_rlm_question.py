#!/usr/bin/env python3
"""
Simple test to verify RLM can answer: "Hoeveel vakantiedagen krijg je bij DSL"
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import sys
import mlflow

# ========================================
# CONFIGURATION - Change your question here
# ========================================

# Test question - modify this to test different questions
TEST_QUESTION = "Hoeveel vakantiedagen krijg je bij DSL & wie is eigenlijk de vertrouwenspersoon?"

# Alternative questions to try (uncomment to use):
# TEST_QUESTION = "Wat is het salaris bij DSL?"
# TEST_QUESTION = "Welke secundaire arbeidsvoorwaarden biedt DSL?"
# TEST_QUESTION = "Hoeveel verlofdagen krijg je bij DSL?"
# TEST_QUESTION = "Wat is de pensioenregeling bij DSL?"

# RLM Configuration
RLM_CONFIG = {
    "max_iterations": 5,  # Limits total recursive calls (depth not implemented yet)
    "enable_logging": True,  # Enable to see agent outputs
    "model": "gpt-5",
    "recursive_model": "gpt-5-nano",
    "depth": 1,  # Currently not implemented in rlm-minimal
    "track_timing": True,  # Track timing for each call
    "print_outputs": True  # Print all agent outputs
}

# Document Configuration
DATA_PATH = "data/files_for_test_set"

# MLflow Configuration
MLFLOW_CONFIG = {
    "enabled": True,
    "experiment_name": "rlm-dsl-test",
    "autolog_openai": True,  # Auto-trace all OpenAI calls
}

# ========================================

# Custom timing and logging wrapper
class TimedRLM:
    """Wrapper to track timing and outputs for RLM calls"""
    
    def __init__(self, rlm_instance, config):
        self.rlm = rlm_instance
        self.config = config
        self.call_times = []
        self.call_outputs = []
        self.total_calls = 0
    
    @mlflow.trace(name="rlm_completion", span_type="LLM")
    def completion(self, context, query):
        """Track RLM completion with timing and output logging"""
        import time
        
        print(f"\n{'='*60}")
        print(f"🤖 RLM CALL #{self.total_calls + 1}")
        print(f"📝 Query: {query}")
        print(f"⚙️  Model: {self.config['model']} (recursive: {self.config['recursive_model']})")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Make the actual RLM call
            response = self.rlm.completion(context=context, query=query)
            
            end_time = time.time()
            call_duration = end_time - start_time
            
            # Store call data
            self.call_times.append(call_duration)
            self.call_outputs.append(response)
            self.total_calls += 1
            
            # Print timing and output info
            print(f"⏱️  Duration: {call_duration:.2f} seconds")
            print(f"📊 Total calls so far: {self.total_calls}")
            print(f"📈 Average duration: {sum(self.call_times)/len(self.call_times):.2f}s")
            
            if self.config.get("print_outputs", True):
                print(f"\n📤 RLM OUTPUT:")
                print("-" * 40)
                print(response)
                print("-" * 40)
            
            print(f"{'='*60}\n")
            
            return response
            
        except Exception as e:
            end_time = time.time()
            call_duration = end_time - start_time
            
            print(f"❌ ERROR after {call_duration:.2f} seconds: {e}")
            print(f"{'='*60}\n")
            
            raise e
    
    def get_stats(self):
        """Get timing statistics"""
        if not self.call_times:
            return {}
        
        return {
            "total_calls": self.total_calls,
            "total_time": sum(self.call_times),
            "avg_time": sum(self.call_times) / len(self.call_times),
            "min_time": min(self.call_times),
            "max_time": max(self.call_times),
            "call_times": self.call_times
        }

# ========================================

# Configure MLflow tracking
if MLFLOW_CONFIG["enabled"]:
    if MLFLOW_CONFIG.get("autolog_openai", True):
        mlflow.openai.autolog()
    mlflow_db = Path(__file__).parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    mlflow.disable_system_metrics_logging()
    print(f"✓ MLflow tracking enabled (SQLite): {mlflow_db}")

# Add rlm-minimal to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'rlm-minimal-main'))

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

try:
    from rlm.rlm_repl import RLM_REPL
    print("✓ RLM_REPL imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RLM_REPL: {e}")
    sys.exit(1)

def load_dsl_documents():
    """Load DSL HR documents"""
    documents = []
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return documents
    
    for file_path in os.listdir(data_path):
        full_path = os.path.join(data_path, file_path)
        if os.path.isfile(full_path):
            try:
                if file_path.endswith('.pdf'):
                    # Handle PDF files
                    import fitz  # PyMuPDF
                    with fitz.open(full_path) as doc:
                        text = ""
                        for page in doc:
                            page_text = page.get_text()
                            # Clean up Unicode characters
                            text += page_text.encode('utf-8', errors='ignore').decode('utf-8') + "\n"
                        documents.append({
                            'filename': file_path,
                            'content': text,
                            'type': 'pdf'
                        })
                elif file_path.endswith(('.txt', '.md', '.json')):
                    # Handle text files
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        documents.append({
                            'filename': file_path,
                            'content': content,
                            'type': 'text'
                        })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents

def prepare_context(documents):
    """Prepare context for RLM in the correct format"""
    # RLM expects either a string, dict, or list of dicts with content
    context_parts = []
    
    for i, doc in enumerate(documents):
        content = doc.get('content', str(doc))
        # Clean up Unicode characters
        clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Create proper document structure for RLM
        context_parts.append({
            "filename": doc.get('filename', f'document_{i+1}'),
            "content": clean_content,
            "type": doc.get('type', 'text')
        })
    
    return context_parts

@mlflow.trace(name="test_rlm_question")
def test_rlm_question():
    """Test RLM with DSL vacation days question"""
    print("🔍 Testing RLM with DSL vacation days question...")

    # Start MLflow run
    if MLFLOW_CONFIG["enabled"]:
        mlflow.start_run(run_name=f"rlm_test_{int(time.time())}")
        mlflow.log_param("question", TEST_QUESTION)
        mlflow.log_param("model", RLM_CONFIG["model"])
        mlflow.log_param("recursive_model", RLM_CONFIG["recursive_model"])
        mlflow.log_param("max_iterations", RLM_CONFIG["max_iterations"])
        mlflow.log_param("depth", RLM_CONFIG["depth"])
        mlflow.log_param("data_path", DATA_PATH)
        print("✓ MLflow run started")

    try:
        # Load documents
        print("📄 Loading DSL documents...")
        doc_load_start = time.time()
        documents = load_dsl_documents()
        doc_load_time = time.time() - doc_load_start

        if not documents:
            print("❌ No documents found!")
            if MLFLOW_CONFIG["enabled"]:
                mlflow.log_param("error", "no_documents_found")
                mlflow.end_run(status="FAILED")
            return False

        print(f"✓ Loaded {len(documents)} documents")

        if MLFLOW_CONFIG["enabled"]:
            mlflow.log_metric("doc_load_time_seconds", doc_load_time)
            mlflow.log_metric("num_documents", len(documents))
            for i, doc in enumerate(documents):
                mlflow.log_param(
                    f"doc_{i}_filename", doc.get('filename', 'unknown')
                )
                mlflow.log_metric(
                    f"doc_{i}_size_chars", len(doc.get('content', ''))
                )

        # Prepare context
        print("� Preparing context...")
        context = prepare_context(documents)
        total_context_size = sum(
            len(str(doc)) for doc in context
        )
        print(f"✓ Context prepared with {len(context)} documents")
        print(f"📊 Total context size: {total_context_size} characters")

        if MLFLOW_CONFIG["enabled"]:
            mlflow.log_metric("total_context_size_chars", total_context_size)

        # Initialize RLM
        print("🤖 Initializing RLM...")
        rlm = RLM_REPL(
            api_key=azure_key,
            model=RLM_CONFIG["model"],
            recursive_model=RLM_CONFIG["recursive_model"],
            max_iterations=RLM_CONFIG["max_iterations"],
            enable_logging=RLM_CONFIG["enable_logging"]
        )
        print("✓ RLM initialized successfully")

        # Wrap with timing tracker
        if RLM_CONFIG.get("track_timing", True):
            rlm = TimedRLM(rlm, RLM_CONFIG)
            print("✓ Timing tracker enabled")

        # Test question
        question = TEST_QUESTION
        print(f"❓ Question: {question}")

        # Get answer from RLM
        print("🧠 Getting answer from RLM...")
        start_time = time.time()

        response = rlm.completion(context=context, query=question)

        total_duration = time.time() - start_time
        print(f"✓ Answer received in {total_duration:.2f} seconds")

        print("\n" + "="*80)
        print("RLM ANSWER:")
        print("="*80)
        print(response)
        print("="*80)

        # Log metrics and artifacts to MLflow
        if MLFLOW_CONFIG["enabled"]:
            mlflow.log_metric("total_response_time_seconds", total_duration)
            mlflow.log_metric("answer_length", len(response) if response else 0)

            # Log question and answer as text artifacts
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(response or "", "answer.txt")

            # Log timing stats from TimedRLM
            if hasattr(rlm, 'get_stats'):
                stats = rlm.get_stats()
                if stats:
                    mlflow.log_metric("rlm_total_calls", stats['total_calls'])
                    mlflow.log_metric("rlm_total_time", stats['total_time'])
                    mlflow.log_metric("rlm_avg_time", stats['avg_time'])
                    mlflow.log_metric("rlm_min_time", stats['min_time'])
                    mlflow.log_metric("rlm_max_time", stats['max_time'])
                    mlflow.log_dict(
                        {"call_times": stats['call_times']},
                        "call_times.json"
                    )

        # Print timing statistics
        if hasattr(rlm, 'get_stats'):
            stats = rlm.get_stats()
            if stats:
                print(f"\n📊 TIMING STATISTICS:")
                print(f"📈 Total RLM calls: {stats['total_calls']}")
                print(f"⏱️  Total time: {stats['total_time']:.2f}s")
                print(f"📊 Average per call: {stats['avg_time']:.2f}s")
                print(f"⚡ Fastest call: {stats['min_time']:.2f}s")
                print(f"🐌 Slowest call: {stats['max_time']:.2f}s")
                print(f"📋 Call times: {[f'{t:.2f}s' for t in stats['call_times']]}")

        if MLFLOW_CONFIG["enabled"]:
            mlflow.end_run()
            print("✓ MLflow run completed")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        if MLFLOW_CONFIG["enabled"]:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        return False

if __name__ == "__main__":
    print("🚀 Starting RLM Question Test")
    print("="*50)
    
    success = test_rlm_question()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
