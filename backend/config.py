import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    google_cse_id: Optional[str] = os.getenv("GOOGLE_CSE_ID")
    bing_search_api_key: Optional[str] = os.getenv("BING_SEARCH_API_KEY")
    serpapi_key: Optional[str] = os.getenv("SERPAPI_KEY")
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    weaviate_url: Optional[str] = os.getenv("WEAVIATE_URL")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
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
    enable_real_time_search: bool = os.getenv("ENABLE_REAL_TIME_SEARCH", "true").lower() == "true"
    enable_image_search: bool = os.getenv("ENABLE_IMAGE_SEARCH", "true").lower() == "true"
    
    # Advanced Features
    enable_code_execution: bool = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true"
    enable_web_browsing: bool = os.getenv("ENABLE_WEB_BROWSING", "true").lower() == "true"
    enable_file_analysis: bool = os.getenv("ENABLE_FILE_ANALYSIS", "true").lower() == "true"
    enable_voice_mode: bool = os.getenv("ENABLE_VOICE_MODE", "true").lower() == "true"
    enable_vision_mode: bool = os.getenv("ENABLE_VISION_MODE", "true").lower() == "true"
    
    # Rate Limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    class Config:
        env_file = ".env"

settings = Settings()