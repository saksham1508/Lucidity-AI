import React from 'react';
import { motion } from 'framer-motion';
import { ExternalLink, Star } from 'lucide-react';
import { SearchResult } from '../types/api';

interface SearchResultCardProps {
  result: SearchResult;
}

export function SearchResultCard({ result }: SearchResultCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-slate-700/50 border border-slate-600 rounded-lg p-4 hover:border-purple-500 transition-colors"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <a
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-purple-400 hover:text-purple-300 font-semibold text-sm mb-2 flex items-center gap-2 group"
          >
            {result.title}
            <ExternalLink size={14} className="group-hover:translate-x-0.5 transition-transform" />
          </a>

          <p className="text-gray-300 text-sm mb-3 line-clamp-2">{result.content}</p>

          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs bg-slate-600 text-slate-200 px-2 py-1 rounded">
              {result.source}
            </span>
            <div className="flex items-center gap-1">
              <Star size={12} className="text-yellow-500 fill-yellow-500" />
              <span className="text-xs text-gray-400">
                {(result.relevance_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
