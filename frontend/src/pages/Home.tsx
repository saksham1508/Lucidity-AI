import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { MessageSquare, Search, History, Zap, Brain, Lock } from 'lucide-react';

export function Home() {
  const features = [
    {
      icon: <MessageSquare size={24} />,
      title: 'Intelligent Chat',
      description: 'Converse with an AI that understands context and personality',
    },
    {
      icon: <Search size={24} />,
      title: 'Smart Search',
      description: 'Multi-source search with citation triangulation',
    },
    {
      icon: <History size={24} />,
      title: 'Memory System',
      description: 'AI that remembers your preferences and history',
    },
    {
      icon: <Zap size={24} />,
      title: 'Fast Generation',
      description: 'RAG-powered responses with real-time information',
    },
    {
      icon: <Brain size={24} />,
      title: 'Adaptive Personality',
      description: 'Switch between assistant, tutor, coder, and more',
    },
    {
      icon: <Lock size={24} />,
      title: 'Privacy First',
      description: 'End-to-end encryption and local-only mode',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-600 to-purple-400 mb-4">
            Lucidity AI
          </h1>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Next-generation AI agent platform with multimodal intelligence, real-time search, and adaptive personality
          </p>
          <Link
            to="/chat"
            className="inline-block px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/50 transition-all"
          >
            Start Chatting
          </Link>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 hover:border-purple-500 transition-colors"
            >
              <div className="text-purple-400 mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-gray-400 text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-20 bg-slate-800/50 border border-slate-700 rounded-lg p-8 text-center"
        >
          <h2 className="text-2xl font-bold text-white mb-4">Ready to experience the future of AI?</h2>
          <p className="text-gray-400 mb-6">Choose your personality and start exploring what Lucidity AI can do</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/chat"
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              Chat Now
            </Link>
            <Link
              to="/profile"
              className="px-6 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
            >
              Customize Profile
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
