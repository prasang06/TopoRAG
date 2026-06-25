"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ExternalLink, ChevronDown, ChevronRight, FileText, Beaker } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface ExtractedPapersProps {
  graphData: {
    nodes: any[];
    links: any[];
  };
}

export default function ExtractedPapers({ graphData }: ExtractedPapersProps) {
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});

  const toggleNode = (id: string) => {
    setExpandedNodes(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // Sort nodes: Structural bottlenecks first, then semantic hits
  const sortedNodes = [...(graphData?.nodes || [])].sort((a, b) => {
    const isBridgeA = a.role.toLowerCase().includes("bridge") || a.role.toLowerCase().includes("bottleneck");
    const isBridgeB = b.role.toLowerCase().includes("bridge") || b.role.toLowerCase().includes("bottleneck");
    if (isBridgeA && !isBridgeB) return -1;
    if (!isBridgeA && isBridgeB) return 1;
    return 0;
  });

  if (!sortedNodes.length) {
    return (
      <div className="w-full flex-1 flex items-center justify-center text-neutral-500 italic bg-black/40 border border-white/10 rounded-2xl">
        No papers extracted for this query.
      </div>
    );
  }

  return (
    <div className="w-full flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar pb-10">
      {sortedNodes.map((node, index) => {
        const isStructural = node.role.toLowerCase().includes("bridge") || node.role.toLowerCase().includes("bottleneck");
        const isExpanded = !!expandedNodes[node.id];

        return (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            key={node.id}
            className={`border rounded-xl overflow-hidden transition-all duration-300 ${isStructural ? "bg-indigo-950/20 border-indigo-500/20 shadow-[0_0_15px_rgba(99,102,241,0.05)]" : "bg-cyan-950/10 border-cyan-500/20 shadow-[0_0_15px_rgba(34,211,238,0.02)]"}`}
          >
            {/* Header / Clickable Area */}
            <div 
              className="p-4 cursor-pointer hover:bg-white/5 transition-colors group flex flex-col gap-2"
              onClick={() => toggleNode(node.id)}
            >
              <div className="flex justify-between items-start gap-4">
                <div className="flex items-center gap-2">
                  {isStructural ? (
                    <div className="bg-indigo-500/20 text-indigo-300 p-1.5 rounded-lg border border-indigo-500/30">
                      <Beaker size={14} />
                    </div>
                  ) : (
                    <div className="bg-cyan-500/20 text-cyan-300 p-1.5 rounded-lg border border-cyan-500/30">
                      <FileText size={14} />
                    </div>
                  )}
                  <h3 className="font-semibold text-neutral-200 text-sm leading-tight group-hover:text-white transition-colors">
                    {node.title}
                  </h3>
                </div>
                
                <div className="flex items-center gap-3 shrink-0 mt-1">
                  <a
                    href={`https://arxiv.org/abs/${node.id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-1 text-[10px] uppercase tracking-widest font-semibold px-2 py-1 rounded-full bg-white/5 hover:bg-cyan-500/20 hover:text-cyan-400 border border-white/10 hover:border-cyan-500/30 transition-all text-neutral-400"
                  >
                    arXiv: {node.id} <ExternalLink size={10} />
                  </a>
                  {isExpanded ? <ChevronDown size={16} className="text-neutral-500" /> : <ChevronRight size={16} className="text-neutral-500" />}
                </div>
              </div>
              
              <div className="flex gap-2 text-xs">
                <span className={`px-2 py-0.5 rounded-md font-medium border ${isStructural ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/20" : "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"}`}>
                  {isStructural ? "Structural Bottleneck" : "Semantic Hit"}
                </span>
                {node.authors && node.authors.length > 0 && (
                  <span className="text-neutral-500 truncate mt-0.5 max-w-[300px]">
                    {node.authors.join(", ")}
                  </span>
                )}
              </div>
            </div>

            {/* Expandable Content */}
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 pt-0 border-t border-white/5 space-y-6">
                    {/* Abstract */}
                    {node.summary && (
                      <div className="mt-4">
                        <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-widest mb-2">Abstract</h4>
                        <div className="text-sm text-neutral-300 leading-relaxed bg-black/40 p-4 rounded-xl border border-white/5 prose prose-invert max-w-none prose-p:leading-relaxed prose-a:text-cyan-400">
                          <ReactMarkdown 
                            remarkPlugins={[remarkMath]} 
                            rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                          >
                            {node.summary.replace(/(?<!\n)\n(?!\n)/g, " ")}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {/* Extracted Branches */}
                    {node.relevant_branches && node.relevant_branches.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-neutral-500 uppercase tracking-widest mb-2">Extracted Anatomy</h4>
                        <div className="space-y-3">
                          {node.relevant_branches.map((branch: any, i: number) => (
                            <div key={i} className="bg-black/20 border border-white/5 rounded-lg overflow-hidden">
                              <div className="bg-white/5 px-3 py-2 text-xs font-medium text-neutral-400 border-b border-white/5 flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-cyan-500/50" />
                                {branch.title}
                              </div>
                              <div className="p-3 space-y-3">
                                {branch.content.map((item: any, j: number) => (
                                  <div key={j} className="text-sm">
                                    {item.type === "equation" ? (
                                      <div className="bg-indigo-950/30 text-indigo-200 p-3 rounded-lg border border-indigo-500/20 overflow-x-auto flex justify-center shadow-inner">
                                        <ReactMarkdown 
                                          remarkPlugins={[remarkMath]} 
                                          rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                                        >
                                          {`$$${item.text}$$`}
                                        </ReactMarkdown>
                                      </div>
                                    ) : (
                                      <div className="text-neutral-300 leading-relaxed prose prose-invert max-w-none prose-sm prose-p:leading-relaxed prose-a:text-cyan-400 prose-code:bg-white/10 prose-code:text-cyan-300 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none">
                                        <ReactMarkdown 
                                          remarkPlugins={[remarkMath]} 
                                          rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                                        >
                                          {item.text
                                            .replace(/(?<!\n)\n(?!\n)/g, " ")
                                            .replace(/\\cite\{[^}]+\}/g, "")
                                            .replace(/\\ref\{[^}]+\}/g, "")
                                            .replace(/\\label\{[^}]+\}/g, "")
                                            .replace(/\\textbf\{([^}]+)\}/g, "**$1**")
                                            .replace(/\\emph\{([^}]+)\}/g, "*$1*")
                                            .replace(/\\begin\{equation\*?\}/g, "\n$$\n")
                                            .replace(/\\end\{equation\*?\}/g, "\n$$\n")
                                          }
                                        </ReactMarkdown>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        );
      })}
    </div>
  );
}
