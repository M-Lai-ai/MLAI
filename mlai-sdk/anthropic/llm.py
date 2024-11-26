import os
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Anthropic_LLM:
    """
    A class to handle interactions with Anthropic's Claude language models.

    This class provides an interface to interact with Anthropic's API for chat
    completions. It handles the configuration and execution of requests to the API.

    Attributes:
        model (str): The Anthropic model to use (e.g., "claude-3-5-sonnet-20241022")
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate
        stream (bool): Whether to stream the response
        top_p (float): Controls diversity via nucleus sampling
        stop_sequences (List[str]): Sequences where the model should stop generating
        api_key (str): Anthropic API key from environment variables

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not found in environment variables
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0,
        max_tokens: int = 1024,
        stream: bool = True,
        top_p: float = 0,
        stop_sequences: Optional[List[str]] = None,
    ):
        """
        Initialize the Anthropic LLM interface.

        Args:
            model (str): The Anthropic model to use
            temperature (float): Controls randomness in output (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            stream (bool): Whether to stream the response
            top_p (float): Controls diversity via nucleus sampling
            stop_sequences (List[str], optional): Sequences where generation should stop
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.top_p = top_p
        self.stop_sequences = stop_sequences or []
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict], system: str) -> requests.Response:
        """
        Make a request to Anthropic's API.

        Args:
            messages (List[Dict]): List of message dictionaries to send to the API
            system (str): System prompt to guide the model's behavior

        Returns:
            requests.Response: The response from the API

        Note:
            The response format depends on the 'stream' parameter:
            - If stream=True, returns a streaming response
            - If stream=False, returns a regular response
        """
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "system": system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop_sequences": self.stop_sequences
        }

        return requests.post(url, headers=headers, json=payload, stream=self.stream)
