"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, FileText, ExternalLink, ChevronDown, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default function MicroDrawer({ activeNode, treeData, onClose }: { activeNode: any, treeData: any, onClose: () => void }) {
  const tree = activeNode ? treeData[activeNode.id] : null;
  const [expandedBranch, setExpandedBranch] = useState<number | null>(0);

  // Reset expanded branch when active node changes
  useEffect(() => {
    setExpandedBranch(0);
  }, [activeNode?.id]);

  return (
    <AnimatePresence>
      {activeNode && (
        <motion.div
          initial={{ x: "100%", opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: "100%", opacity: 0 }}
          transition={{ type: "spring", damping: 25, stiffness: 200 }}
          className="fixed right-0 top-0 h-full w-[600px] backdrop-blur-xl bg-neutral-900/90 border-l border-white/10 z-50 shadow-2xl overflow-y-auto"
        >
          <div className="p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <span className={`text-xs font-mono px-2 py-1 rounded-md mb-2 inline-block ${activeNode.role === "Structural Bottleneck" ? "bg-violet-500/20 text-violet-300" : "bg-cyan-500/20 text-cyan-300"}`}>
                  {activeNode.role}
                </span>
                <h2 className="text-xl font-bold text-white tracking-tight leading-tight">
                  {activeNode.title}
                </h2>
                
                {/* Paper Summary / Abstract */}
                {activeNode.summary && (
                  <div className="mt-4 text-sm text-neutral-300 leading-relaxed bg-black/40 p-4 rounded-xl border border-white/5 prose prose-invert max-w-none prose-p:leading-relaxed prose-a:text-cyan-400">
                    <ReactMarkdown 
                      remarkPlugins={[remarkMath]} 
                      rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
                    >
                      {activeNode.summary.replace(/(?<!\n)\n(?!\n)/g, " ")}
                    </ReactMarkdown>
                  </div>
                )}
                
                {/* arXiv External Link */}
                <a 
                  href={`https://arxiv.org/abs/${activeNode.id}`} 
                  target="_blank" 
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 mt-4 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 rounded-lg text-sm font-medium text-cyan-400 transition-colors"
                >
                  <ExternalLink size={14} />
                  Read Full Paper (arXiv:{activeNode.id})
                </a>
              </div>
              <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition-colors text-white shrink-0">
                <X size={20} />
              </button>
            </div>

            <div className="space-y-3 mt-8">
              <h3 className="text-sm font-bold text-neutral-400 tracking-widest uppercase mb-4">Extracted Anatomy</h3>
              {tree ? (
                (Array.isArray(tree) ? tree : tree.branches || []).map((branch: any, idx: number) => {
                  const isExpanded = expandedBranch === idx;
                  return (
                    <div key={idx} className="bg-black/40 border border-white/5 rounded-xl overflow-hidden transition-all duration-300">
                      <button 
                        onClick={() => setExpandedBranch(isExpanded ? null : idx)}
                        className="w-full p-4 flex items-center justify-between text-left hover:bg-white/5 transition-colors"
                      >
                        <h4 className="text-cyan-400 font-medium flex items-center gap-2">
                          <FileText size={16} />
                          {branch.title}
                        </h4>
                        {isExpanded ? <ChevronDown size={16} className="text-neutral-500" /> : <ChevronRight size={16} className="text-neutral-500" />}
                      </button>
                      
                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div 
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="p-4 pt-0 space-y-4 border-t border-white/5 mt-2">
                              {branch.content.map((item: any, i: number) => (
                                <div key={i} className="text-sm">
                                  {item.type === "equation" ? (
                                    <div className="bg-indigo-950/30 text-indigo-200 p-4 rounded-lg border border-indigo-500/20 overflow-x-auto flex justify-center shadow-inner">
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
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })
              ) : (
                <p className="text-neutral-500 italic p-4 bg-black/20 rounded-xl border border-white/5">No structural tree data extracted for this node.</p>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
