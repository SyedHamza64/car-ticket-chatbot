"""
LLM wrappers for LangChain compatibility.
Supports Groq, Gemini, and Ollama.
"""

import logging
import requests
import time
from typing import List, Dict, Any, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

class GroqLLM(LLM):
    """LangChain LLM wrapper for Groq API."""
    
    model: str = GROQ_MODEL
    api_key: str = GROQ_API_KEY
    temperature: float = 0.2
    max_tokens: int = 1000
    fallback_model: str = "llama-3.3-70b-versatile"

    def _should_fallback(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return (
            "error executing plan" in lower
            or "error finding id" in lower
            or "internal error" in lower
        )
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful Italian car detailing support assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        def _invoke(request_payload: Dict[str, Any]) -> str:
            resp = requests.post(url, headers=headers, json=request_payload, timeout=60)
            body_text = resp.text or ""

            # Handle HTTP errors that still carry useful JSON/text payload details.
            if resp.status_code >= 400:
                try:
                    error_json = resp.json()
                    error_msg = str(error_json.get("error", {}).get("message", "")) or body_text
                except Exception:
                    error_msg = body_text
                raise RuntimeError(error_msg or f"Groq HTTP {resp.status_code}")

            data = resp.json()
            # Defensive parsing for non-standard error payloads.
            if isinstance(data, dict) and "error" in data:
                err = data.get("error", {})
                if isinstance(err, dict):
                    raise RuntimeError(str(err.get("message", "Groq returned error payload")))
                raise RuntimeError(str(err))

            result_text = data["choices"][0]["message"]["content"]
            # Some Groq planner failures are returned as plain text content
            # instead of HTTP errors. Treat those as retryable failures.
            if self._should_fallback(result_text):
                raise RuntimeError(result_text)
            return result_text

        try:
            return _invoke(payload)
        except Exception as e:
            err_text = str(e)
            if self.model != self.fallback_model and self._should_fallback(err_text):
                retry_payload = dict(payload)
                retry_payload["model"] = self.fallback_model
                logger.warning(
                    "Groq model '%s' failed with planner/internal error; retrying with '%s'",
                    self.model,
                    self.fallback_model,
                )
                try:
                    return _invoke(retry_payload)
                except Exception as retry_e:
                    logger.error(f"Groq API retry error: {retry_e}")
                    raise
            # Even on the fallback model, this planner error can be transient.
            # Retry once with a short delay before surfacing the error.
            if self._should_fallback(err_text):
                try:
                    time.sleep(1)
                    return _invoke(payload)
                except Exception as retry_same_model_e:
                    logger.error(f"Groq same-model retry error: {retry_same_model_e}")
                    raise
            logger.error(f"Groq API error: {e}")
            raise


class GeminiLLM(LLM):
    """LangChain LLM wrapper for Google Gemini API."""
    
    model: str = GEMINI_MODEL
    api_key: str = GEMINI_API_KEY
    temperature: float = 0.4
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.model.startswith("models/"):
            model_name = self.model
        else:
            model_name = f"models/{self.model}"
        
        url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate:
                    content = candidate["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        text = content["parts"][0].get("text", "")
                        if text:
                            return text
            
            error_msg = f"Unexpected Gemini response format: {str(result)[:500]}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


class OllamaLLM(LLM):
    """LangChain LLM wrapper for local Ollama."""
    
    model: str = OLLAMA_MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.2
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

def get_llm(provider: str, model: Optional[str] = None) -> LLM:
    """Factory to get the appropriate LLM wrapper."""
    provider = provider.lower()
    
    if provider == "grok" or provider == "groq":
        return GroqLLM(model=model or GROQ_MODEL)
    elif provider == "gemini":
        return GeminiLLM(model=model or GEMINI_MODEL)
    elif provider == "ollama":
        return OllamaLLM(model=model or OLLAMA_MODEL)
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to Groq")
        return GroqLLM(model=model or GROQ_MODEL)
