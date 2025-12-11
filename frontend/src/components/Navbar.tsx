import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, MessageSquare, Search, History, User, Settings, Menu, X, Moon, Sun } from 'lucide-react';
import { useAppContext } from '../context/AppContext';

export function Navbar() {
  const [isOpen, setIsOpen] = React.useState(false);
  const { state, dispatch } = useAppContext();

  const toggleDarkMode = () => {
    dispatch({ type: 'SET_DARK_MODE', payload: !state.isDarkMode });
  };

  const navItems = [
    { icon: MessageSquare, label: 'Chat', path: '/chat' },
    { icon: Search, label: 'Search', path: '/search' },
    { icon: History, label: 'Memory', path: '/memory' },
    { icon: User, label: 'Profile', path: '/profile' },
    { icon: Settings, label: 'Settings', path: '/settings' },
  ];

  return (
    <nav className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link
            to="/"
            className="flex items-center gap-2 font-bold text-xl text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600"
          >
            <Brain size={24} />
            Lucidity AI
          </Link>

          <div className="hidden md:flex items-center gap-1">
            {navItems.map(({ icon: Icon, label, path }) => (
              <Link
                key={path}
                to={path}
                className="p-2 text-gray-400 hover:text-purple-400 hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2"
              >
                <Icon size={18} />
                <span className="text-sm">{label}</span>
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleDarkMode}
              className="p-2 text-gray-400 hover:text-yellow-400 hover:bg-slate-800 rounded-lg transition-colors"
            >
              {state.isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>

            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 text-gray-400 hover:text-purple-400 hover:bg-slate-800 rounded-lg transition-colors"
            >
              {isOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>

        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-t border-slate-800 py-2"
          >
            {navItems.map(({ icon: Icon, label, path }) => (
              <Link
                key={path}
                to={path}
                className="block w-full p-3 text-gray-400 hover:text-purple-400 hover:bg-slate-800 rounded-lg transition-colors flex items-center gap-2"
                onClick={() => setIsOpen(false)}
              >
                <Icon size={18} />
                <span className="text-sm">{label}</span>
              </Link>
            ))}
          </motion.div>
        )}
      </div>
    </nav>
  );
}
