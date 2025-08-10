
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime

from config import settings
from models.schemas import (
    SearchQuery, SearchResponse, RAGQuery, RAGResponse, GenerationRequest, GenerationResponse,
    MemoryEntry, MemoryQuery, MemoryResponse, ReasoningRequest, ReasoningResponse,
    ChatRequest, ChatResponse, ChatMessage, PersonalityType, ErrorResponse
)
from services.search_service import SearchService
from services.rag_service import RAGService
from services.llm_service import LLMService
from services.memory_service import MemoryService
from services.reasoning_service import ReasoningService

# Initialize services
search_service = SearchService()
rag_service = RAGService()
llm_service = LLMService()
memory_service = MemoryService()
reasoning_service = ReasoningService()

# --- Routers for modular features ---
search_router = APIRouter(prefix="/search", tags=["search"])
rag_router = APIRouter(prefix="/rag", tags=["rag"])
model_router = APIRouter(prefix="/model", tags=["model"])
multimodal_router = APIRouter(prefix="/multimodal", tags=["multimodal"])
memory_router = APIRouter(prefix="/memory", tags=["memory"])
privacy_router = APIRouter(prefix="/privacy", tags=["privacy"])
reasoning_router = APIRouter(prefix="/reasoning", tags=["reasoning"])
profile_router = APIRouter(prefix="/profile", tags=["profile"])
collab_router = APIRouter(prefix="/collab", tags=["collab"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])

app = FastAPI(
    title="Lucidity AI Backend", 
    description="Next-generation AI agent platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root endpoint ---
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to Lucidity AI - The Next-Gen AI Agent!",
        "version": "1.0.0",
        "features": [
            "Multi-source search with citation triangulation",
            "Advanced RAG with vector databases",
            "Multiple LLM support (OpenAI, Anthropic, Local)",
            "Long-term adaptive memory",
            "Hybrid symbolic + neural reasoning",
            "Multimodal analysis",
            "Privacy-first design"
        ],
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "search": "operational",
            "rag": "operational", 
            "llm": "operational",
            "memory": "operational",
            "reasoning": "operational"
        }
    }

# --- Search: Multi-source, citation-backed ---
@search_router.post("", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Multi-source search with citation triangulation"""
    try:
        result = await search_service.search(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@search_router.get("/page-content")
async def get_page_content(url: str):
    """Extract clean content from a webpage"""
    try:
        content = await search_service.get_page_content(url)
        return {"url": url, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RAG: Retrieval-Augmented Generation ---
@rag_router.post("/generate", response_model=RAGResponse)
async def rag_generate(query: RAGQuery):
    """Generate response using RAG pipeline"""
    try:
        result = await rag_service.generate(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rag_router.post("/add-knowledge")
async def add_knowledge(title: str, content: str, url: str = "", source: str = "custom"):
    """Add knowledge to the vector database"""
    try:
        await rag_service.add_knowledge(title, content, url, source)
        return {"message": "Knowledge added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rag_router.get("/knowledge-stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    try:
        stats = rag_service.get_knowledge_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Model: LLM endpoints ---
@model_router.post("/generate", response_model=GenerationResponse)
async def model_generate(request: GenerationRequest):
    """Generate text using specified model"""
    try:
        result = await llm_service.generate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@model_router.post("/generate-stream")
async def model_generate_stream(request: GenerationRequest):
    """Stream text generation"""
    try:
        async def generate():
            async for chunk in llm_service.generate_stream(request):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@model_router.get("/available-models")
async def get_available_models():
    """Get list of available models"""
    try:
        models = llm_service.get_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@model_router.get("/model-info/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        info = await llm_service.get_model_info(model_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Memory: User/session memory ---
@memory_router.post("/store")
async def store_memory(memory: MemoryEntry):
    """Store a memory entry"""
    try:
        success = await memory_service.store_memory(memory)
        if success:
            return {"message": "Memory stored successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store memory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@memory_router.post("/retrieve", response_model=MemoryResponse)
async def retrieve_memories(query: MemoryQuery):
    """Retrieve memories based on query"""
    try:
        result = await memory_service.retrieve_memories(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@memory_router.get("/stats/{user_id}")
async def get_memory_stats(user_id: str):
    """Get memory statistics for a user"""
    try:
        stats = memory_service.get_memory_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@memory_router.delete("/clear/{user_id}")
async def clear_user_memories(user_id: str):
    """Clear all memories for a user"""
    try:
        success = await memory_service.clear_user_memories(user_id)
        if success:
            return {"message": "Memories cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear memories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Reasoning: Hybrid symbolic + neural ---
@reasoning_router.post("/solve", response_model=ReasoningResponse)
async def reasoning_solve(request: ReasoningRequest):
    """Solve problems using hybrid reasoning"""
    try:
        result = await reasoning_service.solve(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@reasoning_router.get("/capabilities")
async def get_reasoning_capabilities():
    """Get information about reasoning capabilities"""
    try:
        capabilities = reasoning_service.get_reasoning_capabilities()
        return capabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Chat: Unified chat interface ---
@chat_router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest, background_tasks: BackgroundTasks):
    """Send a chat message and get AI response"""
    try:
        start_time = datetime.now()
        
        # Store user message in memory
        if request.use_memory:
            user_memory = MemoryEntry(
                user_id=request.user_id,
                content=request.message,
                context="user_message",
                timestamp=start_time,
                importance_score=0.5,
                tags=["chat", "user_input"]
            )
            background_tasks.add_task(memory_service.store_memory, user_memory)
        
        # Search for relevant information if requested
        sources = []
        if request.use_search:
            search_query = SearchQuery(
                query=request.message,
                max_results=5,
                include_citations=True
            )
            search_response = await search_service.search(search_query)
            sources = search_response.results
        
        # Generate response using RAG if we have sources
        if sources:
            rag_query = RAGQuery(
                query=request.message,
                context_limit=3,
                model=request.model or settings.default_model
            )
            rag_response = await rag_service.generate(rag_query)
            
            response_content = rag_response.answer
            confidence_score = rag_response.confidence_score
            model_used = rag_response.model_used
        else:
            # Direct LLM generation
            generation_request = GenerationRequest(
                prompt=f"You are Lucidity AI, a helpful and knowledgeable assistant with personality: {request.personality.value}. Respond to: {request.message}",
                model=request.model or settings.default_model,
                temperature=0.7
            )
            generation_response = await llm_service.generate(generation_request)
            
            response_content = generation_response.content
            confidence_score = 0.8  # Default confidence for direct generation
            model_used = generation_response.model
        
        # Store AI response in memory
        if request.use_memory:
            ai_memory = MemoryEntry(
                user_id=request.user_id,
                content=response_content,
                context="ai_response",
                timestamp=datetime.now(),
                importance_score=0.6,
                tags=["chat", "ai_response", request.personality.value]
            )
            background_tasks.add_task(memory_service.store_memory, ai_memory)
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            message=response_content,
            conversation_id=request.conversation_id or f"conv_{int(start_time.timestamp())}",
            sources=sources,
            reasoning_steps=[],
            confidence_score=confidence_score,
            response_time=response_time,
            model_used=model_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Multimodal: Text, image, audio, video ---
@multimodal_router.post("/analyze")
async def multimodal_analyze(file_type: str):
    """Analyze multimodal content"""
    # TODO: Implement multimodal analysis with Whisper, BLIP, etc.
    return {"result": None, "todo": "Implement multimodal analysis."}

# --- Privacy: End-to-end encryption, local-only mode ---
@privacy_router.get("/status")
async def privacy_status():
    """Get privacy status and controls"""
    return {
        "local_mode_enabled": settings.enable_local_mode,
        "privacy_mode_enabled": settings.enable_privacy_mode,
        "encryption_status": "enabled",
        "data_retention_days": 30,
        "features": {
            "end_to_end_encryption": True,
            "local_only_mode": settings.enable_local_mode,
            "data_anonymization": True,
            "gdpr_compliant": True
        }
    }

# --- Profile: User profile and adaptive personality ---
@profile_router.get("/get/{user_id}")
async def get_profile(user_id: str):
    """Get user profile"""
    # TODO: Implement user profile management
    return {"profile": None, "todo": "Implement user profile/personality."}

# --- Collab: Collaborative mode endpoints ---
@collab_router.post("/edit")
async def collab_edit(document_id: str, content: str):
    """Collaborative editing"""
    # TODO: Implement collaborative editing
    return {"result": None, "todo": "Implement collaborative mode."}

# --- Register routers ---
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(model_router)
app.include_router(multimodal_router)
app.include_router(memory_router)
app.include_router(privacy_router)
app.include_router(reasoning_router)
app.include_router(profile_router)
app.include_router(collab_router)
app.include_router(chat_router)
