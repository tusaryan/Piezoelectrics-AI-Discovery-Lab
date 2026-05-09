/**
 * TrainingTerminal — real-time log display styled as a terminal.
 */

"use client";

import { useEffect, useRef } from "react";
import { Terminal as TerminalIcon } from "lucide-react";
import { useTrainingStore, type LogEntry } from "@/lib/store/trainingStore";

const LEVEL_CLASSES: Record<string, string> = {
  info: "log-info",
  warning: "log-warning",
  error: "log-error",
  success: "log-success",
};

function formatTime(ts?: string): string {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return "";
  }
}

export default function TrainingTerminal() {
  const logs = useTrainingStore((s) => s.logs);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div className="training-terminal">
      <div className="terminal-header">
        <TerminalIcon size={14} />
        <span>Training Log</span>
        <span className="terminal-count">{logs.length} lines</span>
      </div>
      <div className="terminal-body">
        {logs.length === 0 ? (
          <div className="terminal-empty">
            Waiting for training to start...
          </div>
        ) : (
          logs.map((log, i) => (
            <LogLine key={i} entry={log} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  const level = entry.level || "info";
  const cls = LEVEL_CLASSES[level] || "log-info";
  const time = formatTime(entry.timestamp);

  return (
    <div className={`terminal-line ${cls}`}>
      {time && <span className="terminal-time">{time}</span>}
      <span className="terminal-msg">{entry.message}</span>
    </div>
  );
}
