import os
import sys
import json
import uuid
from typing import Optional, List, Dict
from datetime import datetime
from cohere_llm import Cohere_LLM

class Cohere_Chatbot:
    """
    A chatbot class that manages conversations using Cohere's language models.

    This class provides functionality to maintain conversation history, manage multiple
    conversations, and interact with the Cohere API through the Cohere_LLM class.

    Attributes:
        _chatbot_counter (int): Class variable to track number of chatbot instances
        llm (Cohere_LLM): Instance of the Cohere LLM interface
        system_prompt (str): The system prompt that defines chatbot behavior
        verbose (bool): Whether to print detailed output
        name (str): Name of the chatbot instance
        conversation_folder (str): Path to store conversation history
        history (List[Dict]): List of conversation messages
    """

    _chatbot_counter = 0

    def __init__(
        self,
        llm: Cohere_LLM,
        system_prompt: str = "You are a helpful assistant.",
        verbose: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize the chatbot instance.

        Args:
            llm (Cohere_LLM): Instance of the Cohere LLM interface
            system_prompt (str): The system prompt that defines chatbot behavior
            verbose (bool): Whether to print detailed output
            name (str, optional): Name for the chatbot instance
        """
        Cohere_Chatbot._chatbot_counter += 1
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.chatbot_id = Cohere_Chatbot._chatbot_counter
        self.name = name or f"cohere_chatbot_{self.chatbot_id}"
        self.conversation_folder = self._create_conversation_folder()
        self.history: List[Dict] = []
        self._initialize_conversation()

    def _create_conversation_folder(self) -> str:
        """
        Create and set up the folder structure for storing conversations.

        Returns:
            str: Path to the chatbot's conversation folder
        """
        base_folder = "conversations"
        chatbot_folder = f"{base_folder}/cohere/{self.name}"
        os.makedirs(chatbot_folder, exist_ok=True)
        
        metadata = {
            "chatbot_id": self.chatbot_id,
            "name": self.name,
            "created_at": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "p": self.llm.p
        }
        
        with open(f"{chatbot_folder}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return chatbot_folder

    def _initialize_conversation(self):
        """Initialize a new conversation with the system prompt."""
        self.conversation_id = str(uuid.uuid4())
        self.history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        self._save_conversation()

    def _save_conversation(self):
        """Save the current conversation history to a JSON file."""
        filename = f"{self.conversation_folder}/conversation_{self.conversation_id}.json"
        
        conversation_data = {
            "conversation_id": self.conversation_id,
            "chatbot_name": self.name,
            "chatbot_id": self.chatbot_id,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "history": self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def start_new_conversation(self):
        """Start a new conversation while maintaining chatbot identity."""
        self._initialize_conversation()
        if self.verbose:
            print(f"\nStarted new conversation with ID: {self.conversation_id}")

    def list_conversations(self) -> List[str]:
        """
        List all saved conversations for this chatbot.

        Returns:
            List[str]: List of conversation filenames
        """
        conversations = [f for f in os.listdir(self.conversation_folder) 
                        if f.startswith('conversation_') and f.endswith('.json')]
        return conversations

    def load_conversation(self, conversation_id: str):
        """
        Load a specific conversation by ID.

        Args:
            conversation_id (str): ID of the conversation to load

        Raises:
            FileNotFoundError: If the conversation file is not found
        """
        filename = f"{self.conversation_folder}/conversation_{conversation_id}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.conversation_id = data["conversation_id"]
                self.history = data["history"]
                if self.verbose:
                    print(f"\nLoaded conversation: {conversation_id}")
        else:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

    def _prepare_messages(self, message: str) -> List[Dict]:
        """
        Prepare messages for API request.

        Args:
            message (str): The user message to process

        Returns:
            List[Dict]: List of formatted messages including conversation history
        """
        self.history.append({
            "role": "user",
            "content": {
                "type": "text",
                "text": message
            }
        })
        return self.history

    def _print_streaming_response(self, content: str):
        """
        Print streaming response in real-time.

        Args:
            content (str): Content chunk to print
        """
        sys.stdout.write(content)
        sys.stdout.flush()

    def __call__(self, message: str) -> str:
        """
        Process a user message and return the chatbot's response.

        Args:
            message (str): The user's input message

        Returns:
            str: The chatbot's response
        """
        messages = self._prepare_messages(message)
        response = self.llm._make_request(messages)

        if self.verbose:
            print(f"\n{self.name} - User: ", message)
            print(f"\n{self.name} - Assistant: ", end="")

        collected_messages = []
        if self.llm.stream:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    try:
                        data = json.loads(line)
                        if data["type"] == "content-delta":
                            content = data["delta"]["message"]["content"]["text"]
                            collected_messages.append(content)
                            if self.verbose:
                                self._print_streaming_response(content)
                    except json.JSONDecodeError:
                        continue

            full_response = "".join(collected_messages)
            if self.verbose:
                print("\n")
        else:
            response_data = response.json()
            full_response = response_data["message"]["content"][0]["text"]
            if self.verbose:
                print(full_response + "\n")

        self.history.append({
            "role": "assistant",
            "content": {
                "type": "text",
                "text": full_response
            }
        })
        self._save_conversation()

        return full_response
