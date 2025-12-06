"""Quick test to verify Gemini API response format"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
load_dotenv()

from src.phase4.rag_pipeline_langchain import GeminiLLM
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

print("Testing Gemini API...")
print(f"Model: {GEMINI_MODEL}")
print(f"API Key present: {bool(GEMINI_API_KEY)}")

try:
    llm = GeminiLLM(model=GEMINI_MODEL, api_key=GEMINI_API_KEY)
    response = llm("Say hello in Italian")
    print(f"\n✅ Success! Response: {response[:100]}...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

