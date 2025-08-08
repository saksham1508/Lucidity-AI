import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    bing_search_api_key: Optional[str] = os.getenv("BING_SEARCH_API_KEY")
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    weaviate_url: Optional[str] = os.getenv("WEAVIATE_URL")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./lucidity.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Features
    enable_local_mode: bool = os.getenv("ENABLE_LOCAL_MODE", "true").lower() == "true"
    enable_privacy_mode: bool = os.getenv("ENABLE_PRIVACY_MODE", "true").lower() == "true"
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview")
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "128000"))
    
    # Search Settings
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "30"))
    enable_web_search: bool = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    enable_academic_search: bool = os.getenv("ENABLE_ACADEMIC_SEARCH", "true").lower() == "true"
    
    class Config:
        env_file = ".env"

settings = Settings()