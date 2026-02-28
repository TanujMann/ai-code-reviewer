from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "AI Code Reviewer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MODEL_BACKEND: str = "demo"
    OPENAI_API_KEY: Optional[str] = None
    MAX_CODE_LENGTH: int = 10000
    RATE_LIMIT: int = 100

    class Config:
        env_file = ".env"

settings = Settings()