import time
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

from config import settings
from models.schemas import RAGQuery, RAGResponse, SearchResult, SearchSource
from .search_service import SearchService
from .llm_service import LLMService

class RAGService:
    def __init__(self):
        self.search_service = SearchService()
        self.llm_service = LLMService()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("lucidity_knowledge")
        except:
            self.collection = self.chroma_client.create_collection(
                name="lucidity_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
    
    async def generate(self, query: RAGQuery) -> RAGResponse:
        """Generate response using RAG pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = await self._retrieve_documents(query.query, query.context_limit)
            
            # Step 2: If no relevant docs found, search the web
            if not relevant_docs:
                search_results = await self._search_and_index(query.query, query.context_limit)
                relevant_docs = search_results
            
            # Step 3: Generate response using retrieved context
            response_content = await self._generate_with_context(
                query.query, 
                relevant_docs, 
                query.model, 
                query.temperature
            )
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(query.query, relevant_docs, response_content)
            
            generation_time = time.time() - start_time
            
            return RAGResponse(
                answer=response_content,
                sources=relevant_docs,
                confidence_score=confidence_score,
                model_used=query.model,
                generation_time=generation_time
            )
            
        except Exception as e:
            raise Exception(f"RAG generation failed: {str(e)}")
    
    async def _retrieve_documents(self, query: str, limit: int) -> List[SearchResult]:
        """Retrieve relevant documents from vector database"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'][0] else 1.0
                    
                    search_result = SearchResult(
                        title=metadata.get('title', 'Retrieved Document'),
                        content=doc,
                        url=metadata.get('url', ''),
                        source=SearchSource(metadata.get('source', 'custom')),
                        relevance_score=1.0 - distance,  # Convert distance to relevance
                        timestamp=metadata.get('timestamp', time.time()),
                        citations=metadata.get('citations', [])
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Document retrieval error: {e}")
            return []
    
    async def _search_and_index(self, query: str, limit: int) -> List[SearchResult]:
        """Search web and index results for future use"""
        try:
            # Search the web
            from models.schemas import SearchQuery
            search_query = SearchQuery(
                query=query,
                sources=[SearchSource.WEB, SearchSource.BING],
                max_results=limit,
                include_citations=True
            )
            
            search_response = await self.search_service.search(search_query)
            
            # Index the results
            await self._index_documents(search_response.results)
            
            return search_response.results
            
        except Exception as e:
            print(f"Search and index error: {e}")
            return []
    
    async def _index_documents(self, documents: List[SearchResult]):
        """Index documents in vector database"""
        try:
            if not documents:
                return
            
            # Prepare data for indexing
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                # Combine title and content for better embeddings
                full_text = f"{doc.title}\n\n{doc.content}"
                texts.append(full_text)
                
                metadata = {
                    'title': doc.title,
                    'url': doc.url,
                    'source': doc.source.value,
                    'relevance_score': doc.relevance_score,
                    'timestamp': doc.timestamp.isoformat(),
                    'citations': doc.citations
                }
                metadatas.append(metadata)
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            print(f"Document indexing error: {e}")
    
    async def _generate_with_context(
        self, 
        query: str, 
        context_docs: List[SearchResult], 
        model: str, 
        temperature: float
    ) -> str:
        """Generate response using retrieved context"""
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_part = f"Source {i} ({doc.source.value}):\nTitle: {doc.title}\nContent: {doc.content[:1000]}...\nURL: {doc.url}\n"
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = f"""You are Lucidity AI, an advanced AI assistant that provides accurate, well-researched answers with proper citations.

Context Information:
{context}

User Question: {query}

Instructions:
1. Provide a comprehensive, accurate answer based on the context provided
2. Include relevant citations using [Source X] format
3. If the context doesn't fully answer the question, acknowledge the limitations
4. Synthesize information from multiple sources when possible
5. Be clear, concise, and helpful

Answer:"""

        # Generate response using LLM
        from models.schemas import GenerationRequest
        generation_request = GenerationRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=2000
        )
        
        response = await self.llm_service.generate(generation_request)
        return response.content
    
    def _calculate_confidence(
        self, 
        query: str, 
        context_docs: List[SearchResult], 
        response: str
    ) -> float:
        """Calculate confidence score for the response"""
        if not context_docs:
            return 0.3  # Low confidence without context
        
        # Factors for confidence calculation
        factors = []
        
        # 1. Number of sources
        num_sources = len(context_docs)
        source_factor = min(num_sources / 3.0, 1.0)  # Normalize to max 1.0
        factors.append(source_factor)
        
        # 2. Average relevance of sources
        avg_relevance = sum(doc.relevance_score for doc in context_docs) / len(context_docs)
        factors.append(avg_relevance)
        
        # 3. Source diversity
        unique_sources = len(set(doc.source for doc in context_docs))
        diversity_factor = min(unique_sources / 2.0, 1.0)  # Normalize to max 1.0
        factors.append(diversity_factor)
        
        # 4. Citation presence
        has_citations = any(doc.citations for doc in context_docs)
        citation_factor = 1.0 if has_citations else 0.7
        factors.append(citation_factor)
        
        # 5. Response length (longer responses often indicate more comprehensive answers)
        response_length_factor = min(len(response) / 1000.0, 1.0)
        factors.append(response_length_factor)
        
        # Calculate weighted average
        weights = [0.25, 0.3, 0.2, 0.15, 0.1]  # Weights for each factor
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(confidence, 1.0)
    
    async def add_knowledge(self, title: str, content: str, url: str = "", source: str = "custom"):
        """Add knowledge to the vector database"""
        try:
            # Create SearchResult object
            search_result = SearchResult(
                title=title,
                content=content,
                url=url,
                source=SearchSource(source),
                relevance_score=1.0,
                timestamp=time.time(),
                citations=[]
            )
            
            # Index the document
            await self._index_documents([search_result])
            
        except Exception as e:
            raise Exception(f"Failed to add knowledge: {str(e)}")
    
    async def search_knowledge(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search existing knowledge base"""
        return await self._retrieve_documents(query, limit)
    
    async def clear_knowledge(self):
        """Clear all knowledge from the database"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection("lucidity_knowledge")
            self.collection = self.chroma_client.create_collection(
                name="lucidity_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Failed to clear knowledge: {str(e)}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_dimension": 384,
                "database_type": "ChromaDB"
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_documents": 0
            }