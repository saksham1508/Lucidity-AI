import React from 'react';
import { motion } from 'framer-motion';
import { User, Bot } from 'lucide-react';

interface MessageCardProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

export function MessageCard({ role, content, timestamp }: MessageCardProps) {
  const isUser = role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
          <Bot size={20} className="text-white" />
        </div>
      )}

      <div
        className={`max-w-sm lg:max-w-md px-4 py-2 rounded-lg ${
          isUser
            ? 'bg-purple-600 text-white rounded-br-none'
            : 'bg-slate-700 text-gray-100 rounded-bl-none'
        }`}
      >
        <p className="text-sm leading-relaxed">{content}</p>
        {timestamp && (
          <span className="text-xs opacity-70 mt-1 block">
            {timestamp.toLocaleTimeString()}
          </span>
        )}
      </div>

      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
          <User size={20} className="text-white" />
        </div>
      )}
    </motion.div>
  );
}
