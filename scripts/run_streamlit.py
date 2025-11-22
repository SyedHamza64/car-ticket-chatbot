#!/usr/bin/env python3
"""
Launch script for Streamlit AI Assistant.
This ensures proper environment setup before starting Streamlit.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Get the streamlit app path (go up one level from scripts/ to root)
    app_path = Path(__file__).parent.parent / "streamlit_app.py"
    
    print("="*80)
    print("🚀 Starting AI Support Assistant")
    print("="*80)
    print(f"\n📍 App will be available at: http://localhost:8501")
    print(f"🤖 Model: llama3.1:8b (or qwen2.5:14b if downloaded)")
    print(f"💾 Vector DB: ChromaDB with 19 tickets + 83 guides")
    print(f"\n⚠️  First load may take 10-20 seconds to initialize")
    print("\n" + "="*80)
    print("Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("👋 Shutting down AI Assistant...")
        print("="*80)
        sys.exit(0)

if __name__ == "__main__":
    main()

