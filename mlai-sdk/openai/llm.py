import os
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAI_LLM:
    """
    A class to handle interactions with OpenAI's language models.

    This class provides an interface to interact with OpenAI's API, specifically
    for chat completions. It handles the configuration and execution of requests
    to the API.

    Attributes:
        model (str): The OpenAI model to use (e.g., "gpt-3.5-turbo")
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        stream (bool): Whether to stream the response
        top_p (float): Controls diversity via nucleus sampling
        frequency_penalty (float): Adjusts frequency of token usage
        presence_penalty (float): Adjusts presence of token usage
        api_key (str): OpenAI API key from environment variables

    Raises:
        ValueError: If OPENAI_API_KEY is not found in environment variables
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = True,
        top_p: float = 0,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        """
        Initialize the OpenAI LLM interface.

        Args:
            model (str): The OpenAI model to use
            temperature (float): Controls randomness in output (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            stream (bool): Whether to stream the response
            top_p (float): Controls diversity via nucleus sampling
            frequency_penalty (float, optional): Adjusts frequency of token usage
            presence_penalty (float, optional): Adjusts presence of token usage
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict]) -> requests.Response:
        """
        Make a request to OpenAI's API.

        Args:
            messages (List[Dict]): List of message dictionaries to send to the API

        Returns:
            requests.Response: The response from the API

        Note:
            The response format depends on the 'stream' parameter:
            - If stream=True, returns a streaming response
            - If stream=False, returns a regular response
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "text"},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream
        }

        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty

        return requests.post(url, headers=headers, json=payload, stream=self.stream)
