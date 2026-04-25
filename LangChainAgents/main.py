import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Add it to a .env file in the LangChainAgents folder or your environment.")

print("✅ LangChainAgents project initialized.")
print("Your OpenAI key is loaded successfully.")

# Add your LangChain agent code here.
