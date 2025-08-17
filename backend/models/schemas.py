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
    GOOGLE = "google"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

class SearchSource(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    CUSTOM = "custom"
    BING = "bing"
    GOOGLE = "google"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SCHOLAR = "scholar"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    REALTIME = "realtime"

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

# Advanced Features Models
class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    environment: str = Field(default="sandbox", description="Execution environment")

class CodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time: float
    language: str
    success: bool

class WebBrowsingRequest(BaseModel):
    url: str = Field(..., description="URL to browse")
    action: str = Field(default="read", description="Action: read, screenshot, interact")
    extract_type: str = Field(default="text", description="What to extract: text, links, images, all")
    wait_time: int = Field(default=5, description="Wait time for page load")

class WebBrowsingResponse(BaseModel):
    content: str
    url: str
    title: str
    links: List[str] = []
    images: List[str] = []
    metadata: Dict[str, Any] = {}
    screenshot_url: Optional[str] = None

class FileAnalysisRequest(BaseModel):
    file_data: Optional[str] = Field(None, description="Base64 encoded file data")
    file_url: Optional[str] = Field(None, description="URL to the file")
    file_type: str = Field(..., description="File type: pdf, docx, xlsx, csv, txt, etc.")
    analysis_type: str = Field(default="summary", description="Analysis type: summary, extract, translate, etc.")

class FileAnalysisResponse(BaseModel):
    content: str
    file_type: str
    analysis_type: str
    metadata: Dict[str, Any] = {}
    extracted_text: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = []

class VoiceRequest(BaseModel):
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    language: str = Field(default="en", description="Audio language")
    task: str = Field(default="transcribe", description="Task: transcribe, translate, analyze")

class VoiceResponse(BaseModel):
    transcription: str
    language: str
    confidence: float
    duration: float
    task: str
    translation: Optional[str] = None

class VisionRequest(BaseModel):
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to image")
    task: str = Field(default="describe", description="Task: describe, ocr, analyze, detect")
    detail_level: str = Field(default="medium", description="Detail level: low, medium, high")

class VisionResponse(BaseModel):
    description: str
    task: str
    confidence: float
    detected_objects: List[Dict[str, Any]] = []
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = {}

class RealTimeSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    time_range: str = Field(default="24h", description="Time range: 1h, 24h, 7d, 30d")
    sources: List[str] = Field(default=["news", "social", "web"], description="Real-time sources")
    location: Optional[str] = Field(None, description="Geographic location filter")

class RealTimeSearchResponse(BaseModel):
    results: List[SearchResult]
    trending_topics: List[str] = []
    sentiment: Optional[str] = None
    time_range: str
    location: Optional[str] = None

class AdvancedChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str
    personality: PersonalityType = PersonalityType.ASSISTANT
    use_search: bool = True
    use_memory: bool = True
    use_web_browsing: bool = False
    use_code_execution: bool = False
    use_file_analysis: bool = False
    use_voice: bool = False
    use_vision: bool = False
    use_real_time: bool = False
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    stream: bool = Field(default=False)
    context_files: List[str] = Field(default=[], description="File IDs for context")
    context_urls: List[str] = Field(default=[], description="URLs for context")

class AdvancedChatResponse(BaseModel):
    message: str
    conversation_id: str
    sources: List[SearchResult] = []
    reasoning_steps: List[ReasoningStep] = []
    confidence_score: float
    response_time: float
    model_used: str
    code_executions: List[CodeExecutionResponse] = []
    web_browsing_results: List[WebBrowsingResponse] = []
    file_analysis_results: List[FileAnalysisResponse] = []
    voice_results: List[VoiceResponse] = []
    vision_results: List[VisionResponse] = []
    real_time_results: List[RealTimeSearchResponse] = []
    citations: List[str] = []
    follow_up_questions: List[str] = []

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)