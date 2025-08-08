import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';

import Navbar from './components/Navbar';
import ChatInterface from './components/ChatInterface';
import SearchInterface from './components/SearchInterface';
import ReasoningInterface from './components/ReasoningInterface';
import MemoryInterface from './components/MemoryInterface';
import SettingsInterface from './components/SettingsInterface';
import LandingPage from './components/LandingPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <Navbar />
        
        <motion.main 
          className="container mx-auto px-4 py-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/chat" element={<ChatInterface />} />
            <Route path="/search" element={<SearchInterface />} />
            <Route path="/reasoning" element={<ReasoningInterface />} />
            <Route path="/memory" element={<MemoryInterface />} />
            <Route path="/settings" element={<SettingsInterface />} />
          </Routes>
        </motion.main>

        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
              border: '1px solid #374151',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
