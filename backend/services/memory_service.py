import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from sentence_transformers import SentenceTransformer
import numpy as np

from config import settings
from models.schemas import MemoryEntry, MemoryQuery, MemoryResponse

class MemoryService:
    def __init__(self):
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
        except:
            print("Warning: Redis not available, using in-memory storage")
            self.redis_client = None
            self.memory_store = {}  # Fallback to in-memory storage
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Memory configuration
        self.max_memories_per_user = 10000
        self.memory_decay_days = 30
        self.importance_threshold = 0.1
    
    async def store_memory(self, memory: MemoryEntry) -> bool:
        """Store a memory entry"""
        try:
            # Generate embedding for semantic search
            embedding = self.embedding_model.encode([memory.content]).tolist()[0]
            
            # Create memory data
            memory_data = {
                "user_id": memory.user_id,
                "content": memory.content,
                "context": memory.context,
                "timestamp": memory.timestamp.isoformat(),
                "importance_score": memory.importance_score,
                "tags": memory.tags,
                "embedding": embedding
            }
            
            # Generate memory ID
            memory_id = f"memory:{memory.user_id}:{int(time.time() * 1000)}"
            
            if self.redis_client:
                # Store in Redis
                self.redis_client.hset(memory_id, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in memory_data.items()
                })
                
                # Add to user's memory index
                user_memories_key = f"user_memories:{memory.user_id}"
                self.redis_client.zadd(user_memories_key, {memory_id: memory.importance_score})
                
                # Set expiration based on importance
                expiration_days = max(1, int(memory.importance_score * self.memory_decay_days))
                self.redis_client.expire(memory_id, expiration_days * 24 * 3600)
                
            else:
                # Store in memory (fallback)
                if memory.user_id not in self.memory_store:
                    self.memory_store[memory.user_id] = []
                
                self.memory_store[memory.user_id].append({
                    "id": memory_id,
                    "data": memory_data
                })
                
                # Limit memory size
                if len(self.memory_store[memory.user_id]) > self.max_memories_per_user:
                    self.memory_store[memory.user_id] = self.memory_store[memory.user_id][-self.max_memories_per_user:]
            
            return True
            
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    async def retrieve_memories(self, query: MemoryQuery) -> MemoryResponse:
        """Retrieve memories based on query"""
        start_time = time.time()
        
        try:
            if query.query:
                # Semantic search
                memories = await self._semantic_search(query)
            else:
                # Get recent memories
                memories = await self._get_recent_memories(query)
            
            # Filter by importance
            filtered_memories = [
                m for m in memories 
                if m.importance_score >= query.min_importance
            ]
            
            # Limit results
            limited_memories = filtered_memories[:query.limit]
            
            query_time = time.time() - start_time
            
            return MemoryResponse(
                memories=limited_memories,
                total_count=len(filtered_memories),
                query_time=query_time
            )
            
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return MemoryResponse(memories=[], total_count=0, query_time=0.0)
    
    async def _semantic_search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Perform semantic search on memories"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query.query]).tolist()[0]
            
            memories = []
            
            if self.redis_client:
                # Get all memories for user from Redis
                user_memories_key = f"user_memories:{query.user_id}"
                memory_ids = self.redis_client.zrevrange(user_memories_key, 0, -1)
                
                for memory_id in memory_ids:
                    memory_data = self.redis_client.hgetall(memory_id)
                    if memory_data:
                        # Calculate similarity
                        stored_embedding = json.loads(memory_data.get('embedding', '[]'))
                        if stored_embedding:
                            similarity = self._cosine_similarity(query_embedding, stored_embedding)
                            
                            # Only include if similarity is above threshold
                            if similarity > 0.3:
                                memory_entry = self._parse_memory_data(memory_data)
                                memory_entry.importance_score = similarity  # Use similarity as relevance
                                memories.append(memory_entry)
            else:
                # Search in-memory storage
                user_memories = self.memory_store.get(query.user_id, [])
                
                for memory_item in user_memories:
                    memory_data = memory_item["data"]
                    stored_embedding = memory_data.get('embedding', [])
                    
                    if stored_embedding:
                        similarity = self._cosine_similarity(query_embedding, stored_embedding)
                        
                        if similarity > 0.3:
                            memory_entry = self._parse_memory_data(memory_data)
                            memory_entry.importance_score = similarity
                            memories.append(memory_entry)
            
            # Sort by relevance (similarity score)
            memories.sort(key=lambda x: x.importance_score, reverse=True)
            
            return memories
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    async def _get_recent_memories(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Get recent memories for a user"""
        try:
            memories = []
            
            if self.redis_client:
                # Get memories from Redis sorted by importance
                user_memories_key = f"user_memories:{query.user_id}"
                memory_ids = self.redis_client.zrevrange(user_memories_key, 0, query.limit * 2)
                
                for memory_id in memory_ids:
                    memory_data = self.redis_client.hgetall(memory_id)
                    if memory_data:
                        memory_entry = self._parse_memory_data(memory_data)
                        memories.append(memory_entry)
            else:
                # Get from in-memory storage
                user_memories = self.memory_store.get(query.user_id, [])
                
                for memory_item in user_memories[-query.limit * 2:]:
                    memory_data = memory_item["data"]
                    memory_entry = self._parse_memory_data(memory_data)
                    memories.append(memory_entry)
            
            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return memories
            
        except Exception as e:
            print(f"Error getting recent memories: {e}")
            return []
    
    def _parse_memory_data(self, memory_data: Dict[str, Any]) -> MemoryEntry:
        """Parse memory data from storage format"""
        return MemoryEntry(
            user_id=memory_data.get('user_id', ''),
            content=memory_data.get('content', ''),
            context=memory_data.get('context', ''),
            timestamp=datetime.fromisoformat(memory_data.get('timestamp', datetime.now().isoformat())),
            importance_score=float(memory_data.get('importance_score', 0.0)),
            tags=json.loads(memory_data.get('tags', '[]')) if isinstance(memory_data.get('tags'), str) else memory_data.get('tags', [])
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    async def update_memory_importance(self, user_id: str, memory_content: str, new_importance: float) -> bool:
        """Update the importance score of a memory"""
        try:
            if self.redis_client:
                # Find memory by content (simplified approach)
                user_memories_key = f"user_memories:{user_id}"
                memory_ids = self.redis_client.zrange(user_memories_key, 0, -1)
                
                for memory_id in memory_ids:
                    memory_data = self.redis_client.hgetall(memory_id)
                    if memory_data.get('content') == memory_content:
                        # Update importance score
                        self.redis_client.hset(memory_id, 'importance_score', str(new_importance))
                        self.redis_client.zadd(user_memories_key, {memory_id: new_importance})
                        return True
            else:
                # Update in-memory storage
                user_memories = self.memory_store.get(user_id, [])
                for memory_item in user_memories:
                    if memory_item["data"]["content"] == memory_content:
                        memory_item["data"]["importance_score"] = new_importance
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error updating memory importance: {e}")
            return False
    
    async def delete_memory(self, user_id: str, memory_content: str) -> bool:
        """Delete a specific memory"""
        try:
            if self.redis_client:
                user_memories_key = f"user_memories:{user_id}"
                memory_ids = self.redis_client.zrange(user_memories_key, 0, -1)
                
                for memory_id in memory_ids:
                    memory_data = self.redis_client.hgetall(memory_id)
                    if memory_data.get('content') == memory_content:
                        # Delete memory
                        self.redis_client.delete(memory_id)
                        self.redis_client.zrem(user_memories_key, memory_id)
                        return True
            else:
                # Delete from in-memory storage
                user_memories = self.memory_store.get(user_id, [])
                for i, memory_item in enumerate(user_memories):
                    if memory_item["data"]["content"] == memory_content:
                        del user_memories[i]
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    async def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a user"""
        try:
            if self.redis_client:
                user_memories_key = f"user_memories:{user_id}"
                memory_ids = self.redis_client.zrange(user_memories_key, 0, -1)
                
                # Delete all memory entries
                if memory_ids:
                    self.redis_client.delete(*memory_ids)
                
                # Delete user index
                self.redis_client.delete(user_memories_key)
            else:
                # Clear from in-memory storage
                if user_id in self.memory_store:
                    del self.memory_store[user_id]
            
            return True
            
        except Exception as e:
            print(f"Error clearing user memories: {e}")
            return False
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        try:
            stats = {
                "total_memories": 0,
                "avg_importance": 0.0,
                "oldest_memory": None,
                "newest_memory": None,
                "top_tags": []
            }
            
            if self.redis_client:
                user_memories_key = f"user_memories:{user_id}"
                memory_count = self.redis_client.zcard(user_memories_key)
                stats["total_memories"] = memory_count
                
                if memory_count > 0:
                    # Get sample of memories for stats
                    memory_ids = self.redis_client.zrange(user_memories_key, 0, min(100, memory_count))
                    
                    importance_scores = []
                    timestamps = []
                    all_tags = []
                    
                    for memory_id in memory_ids:
                        memory_data = self.redis_client.hgetall(memory_id)
                        if memory_data:
                            importance_scores.append(float(memory_data.get('importance_score', 0.0)))
                            timestamps.append(memory_data.get('timestamp', ''))
                            tags = json.loads(memory_data.get('tags', '[]'))
                            all_tags.extend(tags)
                    
                    if importance_scores:
                        stats["avg_importance"] = sum(importance_scores) / len(importance_scores)
                    
                    if timestamps:
                        timestamps = [t for t in timestamps if t]
                        if timestamps:
                            stats["oldest_memory"] = min(timestamps)
                            stats["newest_memory"] = max(timestamps)
                    
                    # Count tag frequency
                    tag_counts = {}
                    for tag in all_tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    
                    stats["top_tags"] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            else:
                # Stats from in-memory storage
                user_memories = self.memory_store.get(user_id, [])
                stats["total_memories"] = len(user_memories)
                
                if user_memories:
                    importance_scores = [m["data"]["importance_score"] for m in user_memories]
                    stats["avg_importance"] = sum(importance_scores) / len(importance_scores)
                    
                    timestamps = [m["data"]["timestamp"] for m in user_memories]
                    stats["oldest_memory"] = min(timestamps)
                    stats["newest_memory"] = max(timestamps)
                    
                    all_tags = []
                    for m in user_memories:
                        all_tags.extend(m["data"]["tags"])
                    
                    tag_counts = {}
                    for tag in all_tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    
                    stats["top_tags"] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return stats
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    async def consolidate_memories(self, user_id: str) -> Dict[str, Any]:
        """Consolidate similar memories to reduce redundancy"""
        try:
            # Get all memories for user
            query = MemoryQuery(user_id=user_id, limit=1000)
            response = await self.retrieve_memories(query)
            
            if len(response.memories) < 2:
                return {"consolidated": 0, "message": "Not enough memories to consolidate"}
            
            # Group similar memories
            consolidated_count = 0
            processed_indices = set()
            
            for i, memory1 in enumerate(response.memories):
                if i in processed_indices:
                    continue
                
                similar_memories = [memory1]
                memory1_embedding = self.embedding_model.encode([memory1.content]).tolist()[0]
                
                for j, memory2 in enumerate(response.memories[i+1:], i+1):
                    if j in processed_indices:
                        continue
                    
                    memory2_embedding = self.embedding_model.encode([memory2.content]).tolist()[0]
                    similarity = self._cosine_similarity(memory1_embedding, memory2_embedding)
                    
                    if similarity > 0.8:  # High similarity threshold
                        similar_memories.append(memory2)
                        processed_indices.add(j)
                
                # If we found similar memories, consolidate them
                if len(similar_memories) > 1:
                    consolidated_memory = await self._merge_memories(similar_memories)
                    
                    # Delete old memories
                    for old_memory in similar_memories:
                        await self.delete_memory(user_id, old_memory.content)
                    
                    # Store consolidated memory
                    await self.store_memory(consolidated_memory)
                    consolidated_count += len(similar_memories) - 1
                
                processed_indices.add(i)
            
            return {
                "consolidated": consolidated_count,
                "message": f"Consolidated {consolidated_count} memories"
            }
            
        except Exception as e:
            print(f"Error consolidating memories: {e}")
            return {"error": str(e)}
    
    async def _merge_memories(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """Merge multiple similar memories into one"""
        # Combine content
        combined_content = " | ".join([m.content for m in memories])
        
        # Combine contexts
        combined_context = " | ".join(set([m.context for m in memories if m.context]))
        
        # Use highest importance score
        max_importance = max([m.importance_score for m in memories])
        
        # Combine tags
        all_tags = []
        for m in memories:
            all_tags.extend(m.tags)
        unique_tags = list(set(all_tags))
        
        # Use most recent timestamp
        latest_timestamp = max([m.timestamp for m in memories])
        
        return MemoryEntry(
            user_id=memories[0].user_id,
            content=combined_content,
            context=combined_context,
            timestamp=latest_timestamp,
            importance_score=max_importance,
            tags=unique_tags
        )