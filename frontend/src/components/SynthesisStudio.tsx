"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sparkles, Terminal, ExternalLink } from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

interface SynthesisStudioProps {
  searchId: number;
  query: string;
  mode: string;
  formattedContext: string;
}

export default function SynthesisStudio({ 
  searchId, 
  query, 
  mode, 
  formattedContext 
}: SynthesisStudioProps) {
  const [streamedText, setStreamedText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  
  useEffect(() => {
    if (searchId === 0 || !query) return;
    
    setStreamedText(""); 
    setIsStreaming(true);

    const abortController = new AbortController();

    async function streamSynthesis() {
      try {
        const response = await fetch("http://localhost:8000/api/synthesize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, mode, formatted_context: formattedContext }),
          signal: abortController.signal
        });

        if (!response.body) throw new Error("No response body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          setStreamedText((prev) => prev + chunk);
        }
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error("Stream error:", error);
          setStreamedText((prev) => prev + "\n[Error connecting to local Gemma4 model]");
        }
      } finally {
        setIsStreaming(false);
      }
    }

    streamSynthesis();

    return () => {
      abortController.abort();
    };
  }, [searchId, query, mode, formattedContext]);

  // Fix ChatGPT-style math brackets and convert stubborn plain text citations into links
  const processedText = streamedText
    .replace(/\\\((.*?)\\\)/g, '$$$1$$') // Fix inline math \( \) -> $ $
    .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$') // Fix block math \[ \] -> $$ $$
    .replace(
      /\[arxiv:\s*([a-zA-Z\-0-9./]+)(?:,\s*([^\]]+))?\]/gi,
      (match, id, section) => `[arXiv: ${id}${section ? `, ${section}` : ''}](https://arxiv.org/abs/${id})`
    );

  return (
    <div className="w-full flex-1 min-h-0 bg-black/40 border border-white/10 rounded-2xl flex flex-col overflow-hidden relative">
      <div className="bg-white/5 px-4 py-2 border-b border-white/10 flex items-center gap-2 shrink-0">
        <Terminal size={14} className="text-cyan-400" />
        <span className="text-xs font-medium tracking-widest text-neutral-400 uppercase">Ollama Stream (gemma4:31b)</span>
      </div>
      <div className="p-5 text-sm text-neutral-200 leading-relaxed overflow-y-auto flex-1 prose prose-invert max-w-none">
        <ReactMarkdown 
          remarkPlugins={[remarkMath]} 
          rehypePlugins={[rehypeKatex]}
          components={{
            a: ({ node, ...props }) => (
              <a 
                {...props} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-cyan-400 hover:text-cyan-300 font-medium hover:underline inline-flex items-center gap-1 transition-colors"
              >
                {props.children}
                <ExternalLink size={12} className="ml-0.5" />
              </a>
            )
          }}
        >
          {processedText}
        </ReactMarkdown>
        {isStreaming && <span className="inline-block w-2 h-4 bg-cyan-400 ml-1 animate-pulse align-middle" />}
      </div>
    </div>
  );
}
