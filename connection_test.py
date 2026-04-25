from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from utils.env_loader import load_environment
from langchain_core.messages import HumanMessage

# LLM clients
from langchain_openai import ChatOpenAI


def main():
    print("# Load environment")
    env = load_environment()

    api_keys = {
        "OPENAI_API_KEY": env.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY"),
    }

    print("Loaded environment keys:")
    for name, value in api_keys.items():
        print(f"  {name}: {'FOUND' if value else 'MISSING'}")

    if api_keys["OPENAI_API_KEY"]:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        provider = "OpenAI"
    elif api_keys["ANTHROPIC_API_KEY"]:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-3.5", temperature=0.2)
        provider = "Anthropic"
    else:
        raise ValueError("No supported LLM API key found in environment. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env.")

    print(f"Using provider: {provider}")

    question = "What's the capital of India?"
    print(f"Asking: {question}")

    result = llm.invoke([HumanMessage(content=question)])
    answer = getattr(result, "content", None) or getattr(result, "text", None)

    if answer is None:
        # Fallback to message list if returned structured result
        if hasattr(result, "generations"):
            answer = str(result.generations)

    print("\n--- Response ---")
    print(answer)

    if answer and "new delhi" in answer.lower():
        print("\n✅ Connection Live: the API responded correctly.")
    else:
        print("\n⚠️ Connection returned a response, but the answer may not match expectations.")


if __name__ == "__main__":
    main()