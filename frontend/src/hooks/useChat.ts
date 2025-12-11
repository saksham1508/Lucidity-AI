import { useState, useCallback } from 'react';
import api from '../api/client';
import { ChatRequest, ChatResponse } from '../types/api';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (
      message: string,
      userId: string,
      personality = 'assistant',
      useSearch = true,
      useMemory = true,
      model?: string
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const chatRequest: ChatRequest = {
          message,
          user_id: userId,
          personality,
          use_search: useSearch,
          use_memory: useMemory,
          model,
        };

        const response = await api.post<ChatResponse>('/chat/message', chatRequest);

        setMessages((prev) => [
          ...prev,
          { role: 'user', content: message, timestamp: new Date() },
          {
            role: 'assistant',
            content: response.data.message,
            timestamp: new Date(),
          },
        ]);

        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
  };
}
