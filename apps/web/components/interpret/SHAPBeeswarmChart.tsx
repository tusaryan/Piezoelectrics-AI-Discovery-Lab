"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

interface ShapPoint {
  feature: string;
  mean_abs_shap: number;
  shap_values: number[];
  feature_values: number[];
}

interface SHAPBeeswarmProps {
  data: ShapPoint[];
}

export function SHAPBeeswarmChart({ data }: SHAPBeeswarmProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!data.length || !containerRef.current) return;
    
    // Clear previous
    d3.select(containerRef.current).selectAll("*").remove();
    
    const width = containerRef.current.clientWidth;
    const height = 600;
    const margin = { top: 40, right: 30, bottom: 40, left: 180 };
    
    // Create tooltip div
    const tooltip = d3.select(containerRef.current)
      .append("div")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background-color", "#1e1e2e")
      .style("border", "1px solid #313244")
      .style("color", "#cdd6f4")
      .style("border-radius", "8px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "100")
      .style("box-shadow", "0 4px 12px rgba(0,0,0,0.5)");

    const svg = d3.select(containerRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
      
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Y-Axis (Features)
    const y = d3.scaleBand()
      .range([0, innerHeight])
      .domain(data.map(d => d.feature))
      .padding(1);
      
    // X-Axis (SHAP value)
    const flatShap = data.flatMap(d => d.shap_values);
    const maxShap = d3.max(flatShap.map(Math.abs)) || 1;
    
    const x = d3.scaleLinear()
      .domain([-maxShap * 1.05, maxShap * 1.05])
      .range([0, innerWidth]);
      
    // Color Scale (Feature Value)
    const color = d3.scaleSequential(d3.interpolateRdBu).domain([1, 0]); // High=Red, Low=Blue
    
    // Center line
    svg.append("line")
      .attr("x1", x(0))
      .attr("x2", x(0))
      .attr("y1", 0)
      .attr("y2", innerHeight)
      .attr("stroke", "#4b5563")
      .attr("stroke-dasharray", "4,4")
      .attr("stroke-width", 1);
      
    // Axes
    svg.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5))
      .attr("color", "#9ca3af");
      
    svg.append("g")
      .call(d3.axisLeft(y).tickSize(0))
      .attr("color", "#d1d5db")
      .selectAll("text")
      .attr("font-size", "11px")
      .attr("dx", "-8px");
      
    // Plot points with simple jitter
    data.forEach(featureData => {
       const { feature, shap_values, feature_values } = featureData;
       const yPos = y(feature) || 0;
       
       const points = shap_values.map((v, i) => ({
           x: x(v),
           y: yPos + (Math.random() - 0.5) * 20, // Simple Jitter for performance
           val: feature_values[i],
           rawShap: v
       }));
       
       svg.selectAll(`circle.pt-${feature.replace(/\s+/g, '')}`)
         .data(points)
         .enter()
         .append("circle")
         .attr("cx", d => d.x)
         .attr("cy", d => d.y)
         .attr("r", 3.0)
         .style("fill", d => color(d.val))
         .style("opacity", 0.7)
         .style("cursor", "crosshair")
         .on("mouseover", function(event, d) {
            d3.select(this)
              .transition().duration(100)
              .style("opacity", 1)
              .attr("r", 5.0)
              .style("stroke", "#fff")
              .style("stroke-width", 1.5);
              
            tooltip.style("visibility", "visible")
                   .html(`<div class="font-semibold mb-1 border-b border-[#313244] pb-1">${feature}</div><div>SHAP Value: <span class="font-mono text-emerald-400">${d.rawShap.toFixed(4)}</span></div><div>Feature Value: <span class="font-mono text-amber-400">${d.val.toFixed(3)}</span></div>`);
         })
         .on("mousemove", function(event) {
            const [mx, my] = d3.pointer(event, containerRef.current);
            tooltip.style("top", (my - 10) + "px")
                   .style("left", (mx + 15) + "px");
         })
         .on("mouseout", function() {
            d3.select(this)
              .transition().duration(150)
              .style("opacity", 0.7)
              .attr("r", 3.0)
              .style("stroke", "none");
              
            tooltip.style("visibility", "hidden");
         });
    });
    
    // X-axis label
    svg.append("text")
      .attr("text-anchor", "middle")
      .attr("x", innerWidth/2)
      .attr("y", innerHeight + 35)
      .attr("fill", "#9ca3af")
      .attr("font-size", "12px")
      .text("SHAP value (impact on model output)");
      
  }, [data]);

  return (
    <div className="w-full relative">
       <div ref={containerRef} className="w-full h-[600px] relative" />
       
       {/* Legend */}
       <div className="absolute top-2 right-4 flex flex-col items-end z-10 bg-card/80 p-2 rounded backdrop-blur-sm shadow-sm border">
          <span className="text-xs text-muted-foreground mb-1 font-medium">Feature Value</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-blue-500 font-bold">Low</span>
            <div className="w-24 h-2.5 bg-gradient-to-r from-blue-500 via-purple-500 to-rose-500 rounded-full" />
            <span className="text-[10px] text-rose-500 font-bold">High</span>
          </div>
       </div>
    </div>
  );
}
