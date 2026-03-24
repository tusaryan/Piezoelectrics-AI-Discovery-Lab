"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal } from "lucide-react";

interface LogEntry {
  level: string;
  message: string;
  step: number;
  timestamp: string;
  metadata?: Record<string, number>;
}

interface LogTerminalProps {
  jobId: string | null;
  onComplete?: () => void;
  onEpochLoss?: (epoch: number, loss: number) => void;
}

export function LogTerminal({ jobId, onComplete, onEpochLoss }: LogTerminalProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!jobId) return;

    setLogs([{
      level: "INFO",
      message: `Connected to compute node: ${jobId}...`,
      step: 0,
      timestamp: new Date().toISOString()
    }]);

    const es = new EventSource(`/api/v1/training/logs/${jobId}/stream`);

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'done') {
          es.close();
          onComplete?.();
          return;
        }
        if (data.metadata && typeof data.metadata.epoch === 'number' && typeof data.metadata.loss === 'number') {
          onEpochLoss?.(data.metadata.epoch, data.metadata.loss);
        }
        setLogs(prev => [...prev, data]);
      } catch (e) {
        console.error("SSE parse error", e);
      }
    };

    es.onerror = () => {
      es.close();
      onComplete?.();
    };

    return () => {
      es.close();
    };
  }, [jobId, onComplete, onEpochLoss]);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [logs]);

  return (
    <div className="rounded-xl border border-muted-foreground/20 bg-black/95 overflow-hidden flex flex-col h-[350px] shadow-lg">
      <div className="h-10 bg-zinc-900 border-b border-zinc-800 flex items-center px-4 gap-2 shrink-0">
        <Terminal className="w-4 h-4 text-zinc-400" />
        <span className="text-zinc-400 text-xs font-mono">
          piezo-ai-worker {jobId ? `• [${jobId.substring(0,8)}]` : ''}
        </span>
        <div className="flex-1" />
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed">
        {logs.length === 0 && (
          <div className="text-zinc-600 italic">Waiting for job assignment...</div>
        )}
        
        {logs.map((log, i) => (
          <div key={i} className="mb-1 flex hover:bg-white/5 px-1 rounded transition-colors break-words">
            <span className="text-zinc-500 w-24 shrink-0">
              {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, fractionalSecondDigits: 3 })}
            </span>
            <span className={`w-16 shrink-0 font-semibold ${
              log.level === 'ERROR' ? 'text-red-400' : 
              log.level === 'WARNING' ? 'text-yellow-400' : 
              'text-cyan-400'
            }`}>
              [{log.level}]
            </span>
            <span className="text-zinc-300 ml-2 whitespace-pre-wrap">{log.message}</span>
          </div>
        ))}
        <div ref={bottomRef} className="h-1" />
      </div>
    </div>
  );
}
