"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import SynthesisStudio from "@/components/SynthesisStudio";
import ExtractedPapers from "@/components/ExtractedPapers";
import ControlPanel from "@/components/ControlPanel";

// Dynamically import the graph to avoid SSR window issues
const MacroGraph = dynamic(() => import("@/components/MacroGraph"), { ssr: false });

export default function Dashboard() {
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchId, setSearchId] = useState(0);
  const [activeTab, setActiveTab] = useState<"synthesis" | "papers">("synthesis");
  
  const [graphData, setGraphData] = useState<any>({ nodes: [], links: [] });
  const [treeData, setTreeData] = useState<any>({});
  const [formattedContext, setFormattedContext] = useState("");
  const [currentQuery, setCurrentQuery] = useState("");
  const [currentMode, setCurrentMode] = useState("");

  const handleSearch = async (query: string, mode: string, limit: number) => {
    setIsSearching(true);
    setCurrentQuery(query);
    setCurrentMode(mode);
    setActiveTab("synthesis"); // Reset to synthesis tab on new search

    try {
      const response = await fetch("http://localhost:8000/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode, limit })
      });
      
      if (!response.ok) throw new Error("Search failed");
      
      const data = await response.json();
      setGraphData(data.graph);
      setTreeData(data.trees);
      setFormattedContext(data.formatted_context);
      
      setHasSearched(true);
      setSearchId(s => s + 1); // Trigger LLM stream
    } catch (error) {
      console.error(error);
      alert("Failed to connect to backend engine. Ensure python api.py is running!");
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-neutral-950 font-sans flex justify-center">
      
      {/* Animated Background Orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[50%] bg-indigo-900/20 rounded-full blur-[120px] mix-blend-screen animate-blob" />
        <div className="absolute top-[20%] right-[-10%] w-[35%] h-[40%] bg-cyan-900/20 rounded-full blur-[120px] mix-blend-screen animate-blob animation-delay-2000" />
        <div className="absolute bottom-[-20%] left-[20%] w-[50%] h-[50%] bg-violet-900/20 rounded-full blur-[120px] mix-blend-screen animate-blob animation-delay-4000" />
      </div>

      {/* CENTER COLUMN: Controls & Chat Stream */}
      <div className="w-full max-w-4xl h-full flex flex-col p-6 z-10 bg-neutral-950/60 relative shadow-2xl backdrop-blur-xl border-x border-white/5">
        
        {!hasSearched && !isSearching && (
          <div className="flex-1 flex flex-col items-center justify-center pointer-events-none mb-10">
            <div className="w-32 h-32 bg-cyan-500 rounded-full blur-[100px] opacity-20 animate-pulse absolute" />
            <h1 className="text-4xl font-bold tracking-widest text-white relative z-10">SYNTHESIS STUDIO</h1>
          </div>
        )}

        <ControlPanel onSearch={handleSearch} isSearching={isSearching} />
        
        {hasSearched && !isSearching && (
          <div className="flex-1 min-h-0 animate-in fade-in slide-in-from-bottom-4 duration-700 flex flex-col mt-4">
            {/* Tabs Header */}
            <div className="flex gap-4 border-b border-white/10 mb-4 shrink-0 px-2">
              <button
                onClick={() => setActiveTab("synthesis")}
                className={`pb-2 text-sm font-semibold tracking-wider uppercase transition-colors relative ${
                  activeTab === "synthesis" ? "text-cyan-400" : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                Synthesis Results
                {activeTab === "synthesis" && (
                  <span className="absolute -bottom-[1px] left-0 w-full h-[2px] bg-cyan-400 rounded-t-full shadow-[0_0_8px_#22d3ee]" />
                )}
              </button>
              <button
                onClick={() => setActiveTab("papers")}
                className={`pb-2 text-sm font-semibold tracking-wider uppercase transition-colors relative ${
                  activeTab === "papers" ? "text-indigo-400" : "text-neutral-500 hover:text-neutral-300"
                }`}
              >
                Extracted Papers ({graphData.nodes?.length || 0})
                {activeTab === "papers" && (
                  <span className="absolute -bottom-[1px] left-0 w-full h-[2px] bg-indigo-400 rounded-t-full shadow-[0_0_8px_#818cf8]" />
                )}
              </button>
            </div>

            {/* Tab Content */}
            <div className={`flex-1 min-h-0 flex flex-col ${activeTab === "synthesis" ? "flex" : "hidden"}`}>
              <SynthesisStudio 
                searchId={searchId}
                query={currentQuery}
                mode={currentMode}
                formattedContext={formattedContext}
              />
            </div>
            
            <div className={`flex-1 min-h-0 flex flex-col ${activeTab === "papers" ? "flex" : "hidden"}`}>
              <ExtractedPapers graphData={graphData} />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
