"""
Example prompt templates for the RLM REPL Client.
"""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

# System prompt for the REPL environment with explicit final answer checking
REPL_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable containing your data. It may be a single string OR a list of document dicts (each with "filename", "content", and "type" keys). Always inspect `context` first to understand its structure.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
4. When context is a list of documents, these helper functions are available:
   - `list_documents()` — prints all documents with their index, filename, type, and size
   - `get_document(index)` — returns the full text content of a document by index
   - `search_documents(keyword)` — searches all documents for a keyword, prints matching snippets with document index and filename

RECOMMENDED STRATEGY for multi-document context:
1. First call `list_documents()` to see what documents are available
2. Use `search_documents(keyword)` to find which documents contain relevant information
3. Use `get_document(i)` to retrieve the full content of relevant documents
4. Feed document content to `llm_query()` to extract specific answers
5. Combine answers from multiple documents into your final response

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query.

Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed multiple documents per sub-LLM query.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, with a multi-document context:
```repl
list_documents()
```
Then inspect and search:
```repl
results = search_documents("vacation days")
doc_text = get_document(results[0][0])
answer = llm_query(f"How many vacation days? Here is the document: {{doc_text}}")
print(answer)
```

For a single-string context, you can chunk and query:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the answer? Here is the chunk: {{chunk}}")
print(answer)
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

def build_system_prompt() -> list[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT
        },
    ]


# Prompt at every step to query root LM to make a decision
USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:""" 
def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {"role": "user", "content": "Based on all the information you have, provide a final answer to the user's query."}
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your context yet. Your next action should be to look through, don't just provide a final answer yet.\n\n"
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    else:
        return {"role": "user", "content": "The history before is your previous interactions with the REPL environment. " + USER_PROMPT.format(query=query)}
