"""
config.py
=========
Central configuration for the backend.
Uses pydantic-settings to load from .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────
    APP_NAME: str = "AI Code Reviewer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # ── Server ───────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "vscode-webview://*",      # VS Code webview
        "*"                         # Dev: allow all
    ]
    
    # ── Model Configuration ───────────────────────
    # Options:
    #   "local"  → use your fine-tuned model (needs GPU)
    #   "openai" → use OpenAI as fallback (needs API key)
    #   "demo"   → use rule-based mock (no GPU needed)
    MODEL_BACKEND: str = "demo"
    
    # Path to your fine-tuned LoRA model
    MODEL_PATH: str = "fine-tuning/models/code-reviewer-lora/final"
    
    # Base model name (must match what you trained on)
    BASE_MODEL_NAME: str = "codellama/CodeLlama-7b-Instruct-hf"
    
    # OpenAI fallback (set MODEL_BACKEND="openai" to use)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # ── Inference Settings ────────────────────────
    MAX_CODE_LENGTH: int = 4000    # Characters
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.1       # Low = more deterministic
    
    # ── Rate Limiting ─────────────────────────────
    RATE_LIMIT: str = "30/minute"  # Per IP
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
