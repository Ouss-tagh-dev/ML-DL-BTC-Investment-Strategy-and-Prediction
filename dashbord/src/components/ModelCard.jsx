import React from "react";
import { motion } from "framer-motion";
import { Star, Info, Zap } from "lucide-react";
import { isFavorite, toggleFavorite } from "../utils/storage";

export default function ModelCard({ model, onPredict, onViewDetails }) {
  const isFav = isFavorite(model.id);
  const isDL = model.id === 'mlp' || model.id === 'lstm' || model.id === 'gru' || model.id === 'lstm_cnn' || model.type?.toLowerCase().includes('deep');

  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="bg-gray-800/40 border border-gray-700/50 backdrop-blur-sm rounded-2xl overflow-hidden flex flex-col h-full group hover:border-blue-500/50 transition-all shadow-lg"
    >
      <div className={`h-1.5 w-full ${isDL ? 'bg-gradient-to-r from-purple-500 to-pink-500' : 'bg-gradient-to-r from-blue-500 to-cyan-500'}`} />

      <div className="p-5 flex-1 flex flex-col">
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center space-x-2">
            <div className={`p-2 rounded-lg ${isDL ? 'bg-purple-500/10 text-purple-400' : 'bg-blue-500/10 text-blue-400'}`}>
              <Zap size={18} />
            </div>
            <div>
              <h3 className="font-bold text-white leading-tight">{model.name || model.id}</h3>
              <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">
                {isDL ? 'Deep Learning' : 'Machine Learning'}
              </span>
            </div>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleFavorite(model.id);
              window.dispatchEvent(new Event('storage')); // Simple reactivity hack
            }}
            className={`transition-colors ${isFav ? 'text-yellow-400' : 'text-gray-600 hover:text-gray-400'}`}
          >
            <Star size={20} fill={isFav ? "currentColor" : "none"} />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-900/40 p-3 rounded-xl border border-gray-700/30">
            <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Accuracy</div>
            <div className="text-lg font-bold text-blue-400">{(model.accuracy ? model.accuracy * 100 : 0).toFixed(1)}%</div>
          </div>
          <div className="bg-gray-900/40 p-3 rounded-xl border border-gray-700/30">
            <div className="text-[10px] text-gray-500 uppercase tracking-tighter mb-1">Status</div>
            <div className={`text-sm font-bold flex items-center space-x-1 ${model.loaded ? 'text-green-400' : 'text-gray-500'}`}>
              <div className={`w-1.5 h-1.5 rounded-full ${model.loaded ? 'bg-green-400' : 'bg-gray-500'}`} />
              <span>{model.loaded ? 'Ready' : 'Standby'}</span>
            </div>
          </div>
        </div>

        <div className="mt-auto flex gap-2">
          <button
            onClick={() => onPredict(model.id)}
            className="flex-1 flex items-center justify-center space-x-2 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-xl transition-all shadow-md active:scale-95"
          >
            <Zap size={16} />
            <span>Predict</span>
          </button>
          <button
            onClick={() => onViewDetails(model.id)}
            className="px-4 flex items-center justify-center bg-gray-700/50 hover:bg-gray-700 text-gray-300 rounded-xl transition-all active:scale-95"
          >
            <Info size={18} />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
