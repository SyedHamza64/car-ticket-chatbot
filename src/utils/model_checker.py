"""Utility to check available Ollama models."""
import ollama
from typing import List, Dict, Optional

def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        result = ollama.list()
        
        # Handle different response formats
        models = []
        if hasattr(result, 'models'):
            # ListResponse object with .models attribute
            models = result.models
        elif isinstance(result, dict) and 'models' in result:
            models = result['models']
        elif isinstance(result, list):
            models = result
        else:
            return []
        
        # Extract model names (filter out 'latest' aliases)
        available = []
        seen_models = set()
        for model in models:
            if hasattr(model, 'model'):
                # Model object with .model attribute
                model_name = model.model
            elif isinstance(model, dict):
                model_name = model.get('name') or model.get('model') or model.get('model_name', '')
            elif isinstance(model, str):
                model_name = model
            else:
                continue
            
            # Skip 'latest' aliases and duplicates
            if model_name and ':latest' not in model_name and model_name not in seen_models:
                available.append(model_name)
                seen_models.add(model_name)
        
        return available
    except Exception as e:
        print(f"Error checking available models: {e}")
        return []

def is_model_available(model_name: str) -> bool:
    """Check if a specific model is available."""
    available = get_available_models()
    return any(model_name in name or name == model_name for name in available)

def find_best_available_model(preferred_models: List[str]) -> Optional[str]:
    """Find the best available model from a list of preferred models."""
    available = get_available_models()
    
    for preferred in preferred_models:
        # Check exact match first
        if preferred in available:
            return preferred
        
        # Check partial match (for variants)
        for model in available:
            if preferred.split(':')[0] in model or model.startswith(preferred.split(':')[0]):
                return model
    
    # Return first available model if none match
    return available[0] if available else None

