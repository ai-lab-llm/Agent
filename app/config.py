from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_id: str = Field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", description="HF model id")
    device_map: str = Field(default="cuda:1", description="transformers device_map value (e.g., 'cuda:0' or 'auto')")
    max_new_tokens: int = 256
    temperature: float = 0.0

    # Database 
    sqlite_uri: str = "sqlite:///app/data/protectee.db"

    # Server
    api_prefix: str = "/api"
    debug: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  
        case_sensitive=False, 
    )

settings = Settings()