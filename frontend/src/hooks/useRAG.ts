import { useState, useCallback } from 'react';
import api from '../api/client';
import { RAGQuery, RAGResponse, SearchResult } from '../types/api';

export function useRAG() {
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);

  const generate = useCallback(
    async (query: string, contextLimit?: number, model?: string, temperature?: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const ragQuery: RAGQuery = {
          query,
          context_limit: contextLimit || 5,
          model: model || 'gpt-4-turbo-preview',
          temperature: temperature || 0.7,
        };

        const response = await api.post<RAGResponse>('/rag/generate', ragQuery);

        setAnswer(response.data.answer);
        setSources(response.data.sources);
        setConfidence(response.data.confidence_score);

        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to generate answer';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const addKnowledge = useCallback(
    async (title: string, content: string, url?: string, source?: string) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await api.post('/rag/add-knowledge', {
          title,
          content,
          url: url || '',
          source: source || 'custom',
        });
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to add knowledge';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const getStats = useCallback(
    async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await api.get('/rag/knowledge-stats');
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to get knowledge stats';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clear = useCallback(() => {
    setAnswer('');
    setSources([]);
    setConfidence(0);
  }, []);

  return {
    answer,
    sources,
    isLoading,
    error,
    confidence,
    generate,
    addKnowledge,
    getStats,
    clear,
  };
}
