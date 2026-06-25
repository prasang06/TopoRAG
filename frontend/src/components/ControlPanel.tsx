"use client";

import React, { useState } from "react";
import { Search, Loader2 } from "lucide-react";

interface ControlPanelProps {
  onSearch: (query: string, mode: string, limit: number) => void;
  isSearching: boolean;
}

export default function ControlPanel({ onSearch, isSearching }: ControlPanelProps) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("curriculum");
  const [limit, setLimit] = useState(10);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSearch(query, mode, limit);
  };

  return (
    <div className="w-full bg-white/5 border border-white/10 rounded-2xl shadow-xl p-5 mb-4">
      <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2 mb-4">
        <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee]" />
        Hierarchical Graph-RAG
      </h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        
        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe your research project or idea..."
            className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-4 pr-12 text-sm text-white placeholder-neutral-500 outline-none focus:border-cyan-400 focus:bg-white/10 transition-all font-medium"
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !query.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/40 disabled:opacity-50 transition-colors"
          >
            {isSearching ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
          </button>
        </div>

        {/* Controls */}
        <div className="flex flex-col xl:flex-row gap-4 items-start xl:items-center justify-between text-xs">
          
          <div className="flex items-center gap-2 w-full xl:w-auto">
            <label className="text-neutral-400 font-medium">Mode:</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              disabled={isSearching}
              className="bg-black/60 border border-white/10 text-white rounded-md px-2 py-1.5 outline-none focus:border-cyan-500 flex-1 xl:flex-none"
            >
              <option value="review">Literature Review</option>
              <option value="curriculum">Reading Curriculum</option>
            </select>
          </div>

          <div className="flex items-center gap-3 w-full xl:w-auto">
            <label className="text-neutral-400 font-medium">Max Papers:</label>
            <input
              type="range"
              min="3"
              max="20"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              disabled={isSearching}
              className="accent-cyan-400 flex-1 xl:flex-none"
            />
            <span className="text-white font-mono bg-white/10 px-2 py-0.5 rounded">{limit}</span>
          </div>

        </div>
      </form>
    </div>
  );
}
