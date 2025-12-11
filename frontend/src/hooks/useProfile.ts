import { useState, useCallback } from 'react';
import api from '../api/client';
import { UserProfile, ProfileUpdateRequest, PersonalityType } from '../types/api';

export function useProfile() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchProfile = useCallback(
    async (userId: string) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await api.get<UserProfile>(`/profile/get/${userId}`);
        if (response.data && response.data.user_id) {
          setProfile(response.data);
          return response.data;
        }
        throw new Error('Invalid profile data');
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch profile';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const updateProfile = useCallback(
    async (userId: string, updates: ProfileUpdateRequest) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await api.put<UserProfile>(`/profile/${userId}`, updates);
        setProfile(response.data);
        return response.data;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to update profile';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const updatePersonality = useCallback(
    async (userId: string, personality: PersonalityType) => {
      if (!profile) return null;
      return updateProfile(userId, { preferred_personality: personality });
    },
    [profile, updateProfile]
  );

  const updatePreferences = useCallback(
    async (userId: string, preferences: Record<string, boolean>) => {
      if (!profile) return null;
      return updateProfile(userId, { privacy_settings: preferences });
    },
    [profile, updateProfile]
  );

  return {
    profile,
    isLoading,
    error,
    fetchProfile,
    updateProfile,
    updatePersonality,
    updatePreferences,
  };
}
