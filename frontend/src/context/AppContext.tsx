import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import { PersonalityType, UserProfile } from '../types/api';

export interface AppState {
  userId: string;
  personality: PersonalityType;
  model: string;
  useSearch: boolean;
  useMemory: boolean;
  isDarkMode: boolean;
  isLoading: boolean;
  notification: {
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  } | null;
  userProfile: UserProfile | null;
}

export type AppAction =
  | { type: 'SET_USER_ID'; payload: string }
  | { type: 'SET_PERSONALITY'; payload: PersonalityType }
  | { type: 'SET_MODEL'; payload: string }
  | { type: 'SET_USE_SEARCH'; payload: boolean }
  | { type: 'SET_USE_MEMORY'; payload: boolean }
  | { type: 'SET_DARK_MODE'; payload: boolean }
  | { type: 'SET_LOADING'; payload: boolean }
  | {
      type: 'SET_NOTIFICATION';
      payload: { message: string; type: 'success' | 'error' | 'info' | 'warning' } | null;
    }
  | { type: 'SET_USER_PROFILE'; payload: UserProfile }
  | { type: 'CLEAR_NOTIFICATION' };

const initialState: AppState = {
  userId: `user_${Math.random().toString(36).substr(2, 9)}`,
  personality: 'assistant',
  model: 'gpt-4-turbo-preview',
  useSearch: true,
  useMemory: true,
  isDarkMode: true,
  isLoading: false,
  notification: null,
  userProfile: null,
};

const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_USER_ID':
      return { ...state, userId: action.payload };
    case 'SET_PERSONALITY':
      return { ...state, personality: action.payload };
    case 'SET_MODEL':
      return { ...state, model: action.payload };
    case 'SET_USE_SEARCH':
      return { ...state, useSearch: action.payload };
    case 'SET_USE_MEMORY':
      return { ...state, useMemory: action.payload };
    case 'SET_DARK_MODE':
      return { ...state, isDarkMode: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_NOTIFICATION':
      return { ...state, notification: action.payload };
    case 'SET_USER_PROFILE':
      return { ...state, userProfile: action.payload };
    case 'CLEAR_NOTIFICATION':
      return { ...state, notification: null };
    default:
      return state;
  }
};

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
}
