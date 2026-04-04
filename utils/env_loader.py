from pathlib import Path
from typing import Dict, Optional
import os
from dotenv import load_dotenv

def load_environment(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_path (Path, optional): Path to .env file. If None, searches in project root.
        
    Returns:
        Dict[str, str]: Dictionary of loaded environment variables
    """
    if env_path is None:
        # Find project root (where .env is located)
        current_path = Path(__file__).parent.parent
        env_path = current_path / '.env'
    
    # Load the .env file
    load_dotenv(dotenv_path=env_path)
    
    # Return loaded variables
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")
    }
