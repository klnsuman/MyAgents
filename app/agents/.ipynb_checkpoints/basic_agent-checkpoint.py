from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from utils.env_loader import load_environment

class BasicAgent:
    """
    A simple agent that can have a conversation using an LLM.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the basic agent.
        
        Args:
            model_name (str): The name of the model to use
        """
        # Load environment variables
        env = load_environment()
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=env["OPENAI_API_KEY"],
            temperature=0.7
        )
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.
        
        Args:
            content (str): The content of the system message
        """
        self.conversation_history.append({"role": "system", "content": content})
        
    def chat(self, user_input: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            str: The agent's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Convert history to LangChain message format
        messages = []
        for msg in self.conversation_history:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        return response.content
