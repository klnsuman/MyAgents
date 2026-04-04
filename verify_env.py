"""
Verify AiResearch environment setup
"""

from importlib.metadata import version, PackageNotFoundError
import sys

print("="*70)
print("AiResearch Environment Verification")
print("="*70)

print(f"\nPython: {sys.version.split()[0]}")
print(f"Environment: {sys.prefix.split('/')[-1]}")

packages = {
    "Core LangChain/LangGraph": [
        'langchain',
        'langchain-core',
        'langchain-community',
        'langgraph',
        'langgraph-checkpoint'
    ],
    "LLM Providers": [
        'langchain-openai',
        'langchain-anthropic',
        'langchain-nvidia-ai-endpoints',
        'langchain-groq'
    ],
    "Vector Stores": [
        'chromadb',
        'faiss-cpu'
    ],
    "Data Science": [
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib'
    ],
    "Deep Learning": [
        'torch',
        'transformers',
        'sentence-transformers'
    ],
    "Utilities": [
        'python-dotenv',
        'jupyter',
        'tqdm'
    ]
}

all_installed = True

for category, pkgs in packages.items():
    print(f"\n{category}:")
    print("-"*70)
    
    for pkg in pkgs:
        try:
            v = version(pkg)
            print(f"  ✓ {pkg:35} {v}")
        except PackageNotFoundError:
            print(f"  ❌ {pkg:35} NOT INSTALLED")
            all_installed = False

# Test key imports
print("\nImport Tests:")
print("-"*70)

tests = [
    ("LangChain", "from langchain_core.messages import HumanMessage"),
    ("LangGraph", "from langgraph.graph import StateGraph"),
    ("PyTorch", "import torch; assert torch.cuda.is_available() or True"),
    ("Transformers", "from transformers import AutoTokenizer"),
    ("ChromaDB", "import chromadb"),
]

for name, code in tests:
    try:
        exec(code)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        all_installed = False

print("\n" + "="*70)
if all_installed:
    print("✅ AiResearch environment is ready!")
else:
    print("⚠️ Some packages missing - review above")
print("="*70)