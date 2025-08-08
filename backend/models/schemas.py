from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class PersonalityType(str, Enum):
    ASSISTANT = "assistant"
    TUTOR = "tutor"
    CODER = "coder"
    THERAPIST = "therapist"
    RESEARCHER = "researcher"
    CREATIVE = "creative"

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

class SearchSource(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    CUSTOM = "custom"
    BING = "bing"

# Search Models
class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query")
    sources: List[SearchSource] = Field(default=[SearchSource.WEB], description="Search sources to use")
    max_results: int = Field(default=10, description="Maximum number of results")
    include_citations: bool = Field(default=True, description="Include citation information")

class SearchResult(BaseModel):
    title: str
    content: str
    url: str
    source: SearchSource
    relevance_score: float
    timestamp: datetime
    citations: List[str] = []

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float
    sources_used: List[SearchSource]

# RAG Models
class RAGQuery(BaseModel):
    query: str = Field(..., description="The query for RAG generation")
    context_limit: int = Field(default=5, description="Number of context documents to retrieve")
    model: str = Field(default="gpt-4-turbo-preview", description="Model to use for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")

class RAGResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    confidence_score: float
    model_used: str
    generation_time: float

# Model Generation
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt for generation")
    model: str = Field(default="gpt-4-turbo-preview", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=4000)
    stream: bool = Field(default=False, description="Stream the response")

class GenerationResponse(BaseModel):
    content: str
    model: str
    tokens_used: int
    generation_time: float
    finish_reason: str

# Multimodal Models
class MultimodalRequest(BaseModel):
    file_type: str = Field(..., description="Type of file: image, audio, video, text")
    file_data: Optional[str] = Field(None, description="Base64 encoded file data")
    file_url: Optional[str] = Field(None, description="URL to the file")
    analysis_type: str = Field(default="general", description="Type of analysis to perform")

class MultimodalResponse(BaseModel):
    analysis: str
    file_type: str
    confidence_score: float
    metadata: Dict[str, Any] = {}
    processing_time: float

# Memory Models
class MemoryEntry(BaseModel):
    user_id: str
    content: str
    context: str
    timestamp: datetime
    importance_score: float = Field(ge=0.0, le=1.0)
    tags: List[str] = []

class MemoryQuery(BaseModel):
    user_id: str
    query: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)

class MemoryResponse(BaseModel):
    memories: List[MemoryEntry]
    total_count: int
    query_time: float

# User Profile Models
class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    preferred_personality: PersonalityType = PersonalityType.ASSISTANT
    preferred_model: str = "gpt-4-turbo-preview"
    privacy_settings: Dict[str, bool] = {
        "store_conversations": True,
        "use_web_search": True,
        "share_analytics": False,
        "local_only_mode": False
    }
    learning_preferences: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    preferred_personality: Optional[PersonalityType] = None
    preferred_model: Optional[str] = None
    privacy_settings: Optional[Dict[str, bool]] = None
    learning_preferences: Optional[Dict[str, Any]] = None

# Reasoning Models
class ReasoningRequest(BaseModel):
    problem: str = Field(..., description="The problem to solve")
    reasoning_type: str = Field(default="hybrid", description="Type of reasoning: symbolic, neural, hybrid")
    show_steps: bool = Field(default=True, description="Show reasoning steps")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

class ReasoningStep(BaseModel):
    step_number: int
    description: str
    reasoning_type: str
    confidence: float
    intermediate_result: Optional[str] = None

class ReasoningResponse(BaseModel):
    solution: str
    reasoning_steps: List[ReasoningStep]
    confidence_score: float
    reasoning_time: float
    method_used: str

# Collaboration Models
class CollaborationRequest(BaseModel):
    document_id: str
    content: str
    operation: str = Field(..., description="Operation: edit, comment, suggest, review")
    user_id: str
    position: Optional[Dict[str, int]] = None  # For cursor position, line numbers, etc.

class CollaborationResponse(BaseModel):
    document_id: str
    updated_content: str
    suggestions: List[str] = []
    conflicts: List[str] = []
    timestamp: datetime

# Chat Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user, assistant, system")
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str
    personality: PersonalityType = PersonalityType.ASSISTANT
    use_search: bool = True
    use_memory: bool = True
    model: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    sources: List[SearchResult] = []
    reasoning_steps: List[ReasoningStep] = []
    confidence_score: float
    response_time: float
    model_used: str

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)