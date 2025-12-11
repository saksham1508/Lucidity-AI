import { useState, useCallback } from 'react';
import api from '../api/client';
import { MemoryEntry, MemoryQuery, MemoryResponse } from '../types/api';

export function useMemory() {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const retrieve = useCallback(
    async (userId: string, query?: string, limit?: number, minImportance?: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const memoryQuery: MemoryQuery = {
          user_id: userId,
          query,
          limit: limit || 10,
          min_importance: minImportance || 0,
        };

        const response = await api.post<MemoryResponse>('/memory/retrieve', memoryQuery);
        setMemories(response.data.memories);
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to retrieve memories';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const store = useCallback(
    async (entry: MemoryEntry) => {
      setIsLoading(true);
      setError(null);

      try {
        await api.post('/memory/store', entry);
        setMemories((prev) => [entry, ...prev]);
        return true;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to store memory';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clear = useCallback(
    async (userId: string) => {
      setIsLoading(true);
      setError(null);

      try {
        await api.delete(`/memory/clear/${userId}`);
        setMemories([]);
        return true;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to clear memories';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const getStats = useCallback(
    async (userId: string) => {
      try {
        const response = await api.get(`/memory/stats/${userId}`);
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to get memory stats';
        setError(errorMessage);
        throw err;
      }
    },
    []
  );

  return {
    memories,
    isLoading,
    error,
    retrieve,
    store,
    clear,
    getStats,
  };
}
