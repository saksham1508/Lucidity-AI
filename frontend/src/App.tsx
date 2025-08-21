import React, { useEffect, useState } from 'react';
import api from './api/client';

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('Checking...');

  useEffect(() => {
    // Call FastAPI health endpoint through Vite proxy
    api.get('/health')
      .then((res) => setBackendStatus(`Backend: ${res.data.status}`))
      .catch(() => setBackendStatus('Backend: unreachable'));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-white text-center">
          Welcome to Lucidity AI
        </h1>
        <p className="text-lg text-gray-300 text-center mt-4">
          Your intelligent assistant for enhanced productivity
        </p>
        <div className="mt-6 text-center">
          <span className="inline-block rounded bg-slate-800/60 px-4 py-2 text-slate-200">
            {backendStatus}
          </span>
        </div>
      </div>
    </div>
  );
}

export default App;
