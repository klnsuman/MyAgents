"""
Quick test of AiResearch environment
"""

print("Testing AiResearch environment...")

# Test 1: LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
print("✓ LangChain imports")

# Test 2: LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
print("✓ LangGraph imports")

# Test 3: PyTorch
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# Test 4: Transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✓ Transformers (loaded BERT tokenizer)")

# Test 5: Data Science
import numpy as np
import pandas as pd
import sklearn
print("✓ NumPy, Pandas, Scikit-learn")

print("\n✅ All tests passed! AiResearch environment is ready!")