import { useState, useCallback } from 'react';
import api from '../api/client';
import { SearchQuery, SearchResponse, SearchResult } from '../types/api';

export function useSearch() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');

  const search = useCallback(
    async (searchQuery: string, sources?: string[], maxResults?: number) => {
      setIsLoading(true);
      setError(null);
      setQuery(searchQuery);

      try {
        const request: SearchQuery = {
          query: searchQuery,
          sources: sources || ['web'],
          max_results: maxResults || 10,
          include_citations: true,
        };

        const response = await api.post<SearchResponse>('/search', request);
        setResults(response.data.results);
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Search failed';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clearResults = useCallback(() => {
    setResults([]);
    setQuery('');
  }, []);

  return {
    results,
    isLoading,
    error,
    query,
    search,
    clearResults,
  };
}
