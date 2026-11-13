"""Configuration management for the lesson content generator."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Central configuration class."""
    # LLM Provider: 'gemini', 'openai', or 'mock'
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Default Models
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Loop Termination Boundary: max 1-2 retries
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

    # Default Topic for submission
    DEFAULT_TOPIC = "Introduction to RAG (Retrieval-Augmented Generation)"

    # Persistence Paths
    RUN_HISTORY_PATH = MEMORY_DIR / "run_history.json"
    EVOLVED_GUIDELINES_PATH = MEMORY_DIR / "evolved_guidelines.json"
    FINAL_LESSON_OUTPUT_PATH = OUTPUTS_DIR / "rag_lesson_final.md"
    REJECTION_LOG_OUTPUT_PATH = OUTPUTS_DIR / "rag_rejection_log.json"

    @classmethod
    def get_effective_provider(cls) -> str:
        """Determines active provider based on availability of keys or explicit setting."""
        if cls.LLM_PROVIDER == "mock":
            return "mock"
        if cls.LLM_PROVIDER == "gemini" and cls.GEMINI_API_KEY:
            return "gemini"
        if cls.LLM_PROVIDER == "openai" and cls.OPENAI_API_KEY:
            return "openai"
        # If gemini key exists, default to gemini
        if cls.GEMINI_API_KEY:
            return "gemini"
        # If openai key exists, default to openai
        if cls.OPENAI_API_KEY:
            return "openai"
        # Fallback to mock mode if no active API keys are found
        return "mock"

