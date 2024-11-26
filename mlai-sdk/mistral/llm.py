import os
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Mistral_LLM:
    """
    A class to handle interactions with Mistral AI's language models.

    This class provides an interface to interact with Mistral's API for chat
    completions. It handles the configuration and execution of requests to the API.

    Attributes:
        model (str): The Mistral model to use (e.g., "mistral-large-latest")
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        stream (bool): Whether to stream the response
        top_p (float): Controls diversity via nucleus sampling
        api_key (str): Mistral API key from environment variables

    Raises:
        ValueError: If MISTRAL_API_KEY is not found in environment variables
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        temperature: float = 0.7,
        max_tokens: int = 100,
        stream: bool = True,
        top_p: float = 1,
    ):
        """
        Initialize the Mistral LLM interface.

        Args:
            model (str): The Mistral model to use
            temperature (float): Controls randomness in output (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            stream (bool): Whether to stream the response
            top_p (float): Controls diversity via nucleus sampling
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.top_p = top_p
        self.api_key = os.getenv('MISTRAL_API_KEY')
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict]) -> requests.Response:
        """
        Make a request to Mistral's API.

        Args:
            messages (List[Dict]): List of message dictionaries to send to the API

        Returns:
            requests.Response: The response from the API

        Note:
            The response format depends on the 'stream' parameter:
            - If stream=True, returns a streaming response
            - If stream=False, returns a regular response
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream
        }

        return requests.post(url, headers=headers, json=payload, stream=self.stream)
