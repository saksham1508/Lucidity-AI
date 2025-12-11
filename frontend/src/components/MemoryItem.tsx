import React from 'react';
import { motion } from 'framer-motion';
import { Trash2, Clock, Tag } from 'lucide-react';
import { MemoryEntry } from '../types/api';

interface MemoryItemProps {
  memory: MemoryEntry;
  onDelete?: (id: string) => void;
}

export function MemoryItem({ memory, onDelete }: MemoryItemProps) {
  const date = new Date(memory.timestamp);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-slate-700/50 border border-slate-600 rounded-lg p-4 hover:border-purple-500 transition-colors"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <p className="text-gray-100 text-sm mb-2">{memory.content}</p>

          <div className="flex items-center gap-2 flex-wrap mb-2">
            <div className="flex items-center gap-1 text-gray-400 text-xs">
              <Clock size={12} />
              <span>{date.toLocaleDateString()} {date.toLocaleTimeString()}</span>
            </div>
            {memory.tags.length > 0 && (
              <div className="flex items-center gap-1 flex-wrap">
                {memory.tags.slice(0, 2).map((tag) => (
                  <span
                    key={tag}
                    className="inline-flex items-center gap-1 text-xs bg-purple-600/30 text-purple-300 px-2 py-1 rounded"
                  >
                    <Tag size={10} />
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <div className="flex-1 bg-slate-600 rounded-full h-1 overflow-hidden">
              <div
                className="bg-gradient-to-r from-purple-500 to-pink-500 h-full"
                style={{ width: `${memory.importance_score * 100}%` }}
              />
            </div>
            <span className="text-xs text-gray-400">{(memory.importance_score * 100).toFixed(0)}%</span>
          </div>
        </div>

        {onDelete && (
          <button
            onClick={() => onDelete(memory.user_id)}
            className="text-slate-400 hover:text-red-400 transition-colors"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>
    </motion.div>
  );
}
