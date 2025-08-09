# Lucidity AI

A next-generation AI agent platform with a FastAPI backend and a React + Vite frontend.

## Structure

- `backend/`: FastAPI backend for AI, RAG, and multimodal endpoints
- `frontend/`: React + Vite frontend (TypeScript)
- `shared/`: Shared libraries
- `sdks/`: SDKs for Python, JS, Rust

## Core Features

- **Multi-source Search**: Citation-backed, real-time, multi-source triangulation (web, Bing, academic, custom crawler)
- **RAG Pipeline**: Retrieval-Augmented Generation with vector DBs (Weaviate, Pinecone)
- **Model Integration**: Fine-tuned LLMs (Mixtral, LLaMA 3, open models)
- **Multimodal Endpoints**: Text, image, audio, video (Whisper, BLIP, video captioning)
- **Memory System**: Long-term, user-adaptive memory (Redis, LangChain)
- **Privacy & Security**: End-to-end encryption, local-only mode, OAuth2
- **Reasoning Engine**: Hybrid symbolic + neural reasoning
- **Adaptive Personality**: Switch between assistant, tutor, coder, therapist, etc.
- **Collaborative Mode**: Co-edit documents, codebases, or research papers
- **Offline Intelligence**: Local model fallback
- **Ethical Reasoning**: Built-in bias detection and transparency layer

## Getting Started

### Backend

1. `cd backend`
2. (Recommended) Create a virtual environment
3. `pip install fastapi uvicorn`
4. `uvicorn main:app --reload`

### Frontend

1. `cd frontend`
2. `npm install`
3. `npm run dev`

---

This project is designed for extensibility, modular AI integration, and full multimodal support. See the backend and frontend folders for feature scaffolding and implementation details.
