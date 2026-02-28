from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    mistral_api_key: str = ""
    database_url: str = "sqlite:///./pharma.db"

    model_config = {"env_file": Path(__file__).resolve().parent.parent / ".env"}


settings = Settings()
