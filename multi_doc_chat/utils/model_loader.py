from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any

from dotenv import load_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings

from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# ------------------------------------------------------------------
# API Key Manager
# ------------------------------------------------------------------

class ApiKeyManager:
    """Centralized API key loader with fallback support"""

    REQUIRED_KEYS = {"groq": ["GROQ_API_KEY"], "google": ["GOOGLE_API_KEY"]}

    def __init__(self) -> None:
        self._keys: Dict[str, str] = {}
        self._load_from_env_blob()
        self._load_from_individual_envs()

    def _load_from_env_blob(self) -> None:
        raw = os.getenv("API_KEYS_JSON")
        if not raw:
            return

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("API_KEYS_JSON must be a JSON object")
            self._keys.update(parsed)
            log.info("Loaded API keys from API_KEYS_JSON")
        except Exception as e:
            log.warning("Failed to parse API_KEYS_JSON", error=str(e))

    def _load_from_individual_envs(self) -> None:
        for key in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
            value = os.getenv(key)
            if value:
                self._keys[key] = value

    def validate(self, provider: str) -> None:
        required = self.REQUIRED_KEYS.get(provider, [])
        missing = [k for k in required if not self._keys.get(k)]

        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException(
                f"Missing required API keys: {missing}", sys
            )

    def get(self, key: str) -> str:
        value = self._keys.get(key)
        if not value:
            raise KeyError(f"API key not found: {key}")
        return value


# ------------------------------------------------------------------
# Model Loader
# ------------------------------------------------------------------

class ModelLoader:
    """Factory for Embeddings and LLMs"""

    def __init__(self) -> None:
        self._load_environment()
        self.config = load_config()
        self.api_keys = ApiKeyManager()

        log.info("ModelLoader initialized", config_keys=list(self.config.keys()))

    def _load_environment(self) -> None:
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Local environment detected — .env loaded")
        else:
            log.info("Production environment detected")

    # ---------------------------
    # Embeddings
    # ---------------------------

    def load_embedding(self) -> FastEmbedEmbeddings:
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return FastEmbedEmbeddings(model=model_name)

        except Exception as e:
            log.error("Failed to load embedding model", error=str(e))
            raise DocumentPortalException(
                "Embedding model initialization failed", sys
            )

    # ---------------------------
    # LLM
    # ---------------------------

    def load_llm(self):
        llm_block = self.config.get("llm", {})
        provider_key = os.getenv("LLM_PROVIDER", "groq").lower()

        if provider_key not in llm_block:
            raise ValueError(f"LLM provider not configured: {provider_key}")

        llm_cfg = llm_block[provider_key]
        provider = llm_cfg["provider"]
        model = llm_cfg["model_name"]
        temperature = llm_cfg.get("temperature", 0.2)
        max_tokens = llm_cfg.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model)
        self.api_keys.validate(provider)

        if provider == "groq":
            return self._load_groq(model, temperature)

        if provider == "google":
            return self._load_google(model, temperature, max_tokens)

        raise ValueError(f"Unsupported LLM provider: {provider}")

    # ---------------------------
    # Providers (lazy imports)
    # ---------------------------

    def _load_groq(self, model: str, temperature: float):
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError(
                "langchain-groq not installed. Run: pip install langchain-groq"
            ) from e

        return ChatGroq(
            model=model,
            api_key=self.api_keys.get("GROQ_API_KEY"),
            temperature=temperature,
        )

    def _load_google(
        self, model: str, temperature: float, max_tokens: int
    ):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Run: pip install langchain-google-genai"
            ) from e

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self.api_keys.get("GOOGLE_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )


# ------------------------------------------------------------------
# Local test
# ------------------------------------------------------------------

if __name__ == "__main__":
    loader = ModelLoader()

    embedding = loader.load_embedding()
    print("Embedding loaded")
    print("Embedding sample:", embedding.embed_query("Hello, how are you?")[:5])

    llm = loader.load_llm()
    print("LLM loaded")
    response = llm.invoke("Hello, how are you?")
    print("LLM response:", response.content)
