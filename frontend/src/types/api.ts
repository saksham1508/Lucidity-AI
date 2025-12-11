export type PersonalityType =
  | 'assistant'
  | 'tutor'
  | 'coder'
  | 'therapist'
  | 'researcher'
  | 'creative';

export interface SearchResult {
  title: string;
  content: string;
  url: string;
  source: string;
  relevance_score: number;
  timestamp: string;
  citations: string[];
}

export interface SearchQuery {
  query: string;
  sources?: string[];
  max_results?: number;
  include_citations?: boolean;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  total_results: number;
  search_time: number;
  sources_used: string[];
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  user_id: string;
  personality?: PersonalityType;
  use_search?: boolean;
  use_memory?: boolean;
  model?: string;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  sources: SearchResult[];
  reasoning_steps: any[];
  confidence_score: number;
  response_time: number;
  model_used: string;
}

export interface MemoryEntry {
  user_id: string;
  content: string;
  context: string;
  timestamp: string;
  importance_score: number;
  tags: string[];
}

export interface MemoryQuery {
  user_id: string;
  query?: string;
  limit?: number;
  min_importance?: number;
}

export interface MemoryResponse {
  memories: MemoryEntry[];
  total_count: number;
  query_time: number;
}

export interface UserProfile {
  user_id: string;
  name?: string;
  preferred_personality: PersonalityType;
  preferred_model: string;
  privacy_settings: Record<string, boolean>;
  learning_preferences: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface ProfileUpdateRequest {
  name?: string;
  preferred_personality?: PersonalityType;
  preferred_model?: string;
  privacy_settings?: Record<string, boolean>;
  learning_preferences?: Record<string, any>;
}

export interface RAGQuery {
  query: string;
  context_limit?: number;
  model?: string;
  temperature?: number;
}

export interface RAGResponse {
  answer: string;
  sources: SearchResult[];
  confidence_score: number;
  model_used: string;
  generation_time: number;
}
