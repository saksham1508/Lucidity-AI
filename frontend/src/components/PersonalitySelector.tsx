import React from 'react';
import { motion } from 'framer-motion';
import { Brain, BookOpen, Code, Heart, Lightbulb, Wand2 } from 'lucide-react';
import { PersonalityType } from '../types/api';

interface PersonalitySelectorProps {
  selected: PersonalityType;
  onChange: (personality: PersonalityType) => void;
}

const personalities: Array<{
  id: PersonalityType;
  label: string;
  icon: React.ReactNode;
  description: string;
}> = [
  { id: 'assistant', label: 'Assistant', icon: <Brain size={20} />, description: 'Helpful and balanced' },
  { id: 'tutor', label: 'Tutor', icon: <BookOpen size={20} />, description: 'Educational focus' },
  { id: 'coder', label: 'Coder', icon: <Code size={20} />, description: 'Technical expert' },
  { id: 'therapist', label: 'Therapist', icon: <Heart size={20} />, description: 'Empathetic listener' },
  { id: 'researcher', label: 'Researcher', icon: <Lightbulb size={20} />, description: 'Deep analysis' },
  { id: 'creative', label: 'Creative', icon: <Wand2 size={20} />, description: 'Imaginative ideas' },
];

export function PersonalitySelector({ selected, onChange }: PersonalitySelectorProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      {personalities.map((personality) => (
        <motion.button
          key={personality.id}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => onChange(personality.id)}
          className={`p-3 rounded-lg border transition-all ${
            selected === personality.id
              ? 'border-purple-500 bg-purple-600/20 text-purple-400'
              : 'border-slate-600 bg-slate-700/50 text-gray-400 hover:border-slate-500'
          }`}
        >
          <div className="flex flex-col items-center gap-1">
            <div className="text-current">{personality.icon}</div>
            <span className="text-xs font-medium">{personality.label}</span>
            <span className="text-xs opacity-70">{personality.description}</span>
          </div>
        </motion.button>
      ))}
    </div>
  );
}
