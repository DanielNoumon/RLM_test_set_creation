"""
Thin wrapper for Azure OpenAI / OpenAI chat completions.
"""
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Chat completion client supporting Azure OpenAI and OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-12-01-preview",
    ):
        self.model = model
        self.api_key = api_key or os.getenv(
            "AZURE_OPENAI_API_KEY"
        ) or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set AZURE_OPENAI_API_KEY "
                "or OPENAI_API_KEY in .env"
            )

        self.azure_endpoint = azure_endpoint or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )
        self.azure_api_version = azure_api_version

        if self.azure_endpoint:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)

    def completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a chat completion."""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
