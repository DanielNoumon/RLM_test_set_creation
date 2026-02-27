"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        # Check if using Azure OpenAI
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Use Azure OpenAI if endpoint is provided
        if self.azure_endpoint:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            create_kwargs = {
                "model": self.model,
                "messages": messages,
                **kwargs
            }
            if max_tokens is not None:
                create_kwargs["max_completion_tokens"] = max_tokens
            response = self.client.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")