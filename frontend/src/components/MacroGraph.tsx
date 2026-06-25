"use client";

import React, { useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { mockGraphData } from "@/lib/mockData";

export default function MacroGraph({ onNodeClick, graphData }: { onNodeClick: (node: any) => void, graphData: any }) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = React.useState({ width: 800, height: 600 });

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      for (let entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // Adding a small timeout to let the graph stabilize then center
    setTimeout(() => {
      if (fgRef.current) {
        // Fix extreme zooming by locking to a reasonable max scale if there are very few nodes
        fgRef.current.zoomToFit(400, 50, (node: any) => true);
        
        // Ensure it doesn't zoom in crazy far
        const currentZoom = fgRef.current.zoom();
        if (currentZoom > 2) {
          fgRef.current.zoom(2, 400);
        }
      }
    }, 500);
  }, [graphData]);

  return (
    <div ref={containerRef} className="w-full h-full absolute inset-0 z-0 bg-neutral-950">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeColor={(node: any) => 
          node.role === "Structural Bottleneck" ? "#8b5cf6" : "#06b6d4" // Violet vs Cyan
        }
        nodeRelSize={8}
        linkColor={() => "rgba(255,255,255,0.2)"}
        linkWidth={(link: any) => link.weight || 1}
        onNodeClick={onNodeClick}
        backgroundColor="#0a0a0a"
        // Custom canvas drawing to render text on nodes
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          // Draw the circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.role === "Structural Bottleneck" ? "#8b5cf6" : "#06b6d4";
          ctx.fill();

          // Only draw text if we are sufficiently zoomed in to avoid massive text overlaps
          if (globalScale > 1.5) {
            const label = node.title;
            const fontSize = 12 / globalScale;
            ctx.font = `${fontSize}px Inter, Sans-Serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.fillText(label, node.x, node.y + 8);
          }
        }}
      />
    </div>
  );
}
