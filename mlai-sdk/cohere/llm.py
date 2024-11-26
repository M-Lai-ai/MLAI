import os
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Cohere_LLM:
    """
    A class to handle interactions with Cohere's language models.

    This class provides an interface to interact with Cohere's API for chat
    completions. It handles the configuration and execution of requests to the API.

    Attributes:
        model (str): The Cohere model to use (e.g., "command-r-plus-08-2024")
        temperature (float): Controls randomness in the output (0.0 to 1.0)
        stream (bool): Whether to stream the response
        p (float): Controls nucleus sampling
        frequency_penalty (float): Adjusts frequency of token usage
        presence_penalty (float): Adjusts presence of token usage
        api_key (str): Cohere API key from environment variables

    Raises:
        ValueError: If COHERE_API_KEY is not found in environment variables
    """

    def __init__(
        self,
        model: str = "command-r-plus-08-2024",
        temperature: float = 0.7,
        stream: bool = True,
        p: float = 0,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        """
        Initialize the Cohere LLM interface.

        Args:
            model (str): The Cohere model to use
            temperature (float): Controls randomness in output (0.0 to 1.0)
            stream (bool): Whether to stream the response
            p (float): Controls nucleus sampling
            frequency_penalty (float, optional): Adjusts frequency of token usage
            presence_penalty (float, optional): Adjusts presence of token usage
        """
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_key = os.getenv('COHERE_API_KEY')
        
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

    def _make_request(self, messages: List[Dict]) -> requests.Response:
        """
        Make a request to Cohere's API.

        Args:
            messages (List[Dict]): List of message dictionaries to send to the API

        Returns:
            requests.Response: The response from the API

        Note:
            The response format depends on the 'stream' parameter:
            - If stream=True, returns a streaming response
            - If stream=False, returns a regular response
        """
        url = "https://api.cohere.com/v2/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "p": self.p,
            "stream": self.stream,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

        return requests.post(url, headers=headers, json=payload, stream=self.stream)
