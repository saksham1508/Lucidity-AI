// Simple Axios client with base URL that works with Vite proxy in dev
import axios from 'axios';

const api = axios.create({
  // In dev, vite proxy will map /api -> http://127.0.0.1:8000
  baseURL: '/api',
  timeout: 15000,
});

export default api;