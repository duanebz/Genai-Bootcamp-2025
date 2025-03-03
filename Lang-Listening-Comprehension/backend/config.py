import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
