"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Phone, PhoneOff, Volume2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface VoiceChatProps {
  enabled: boolean;
  apiUrl: string;
}

export function VoiceChat({ enabled, apiUrl }: VoiceChatProps) {
  const [isListening, setIsListening] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState<string>("idle");
  const [, setTranscript] = useState<string>("");
  const [, setResponseText] = useState<string>("");
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const playAudioChunk = useCallback((base64Data: string) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const ctx = audioContextRef.current;
      const binary = atob(base64Data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const float32 = new Float32Array(bytes.length / 2);
      const view = new DataView(bytes.buffer);
      for (let i = 0; i < float32.length; i++) {
        float32[i] = view.getInt16(i * 2, true) / 32768;
      }
      const buffer = ctx.createBuffer(1, float32.length, 24000);
      buffer.getChannelData(0).set(float32);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start();
    } catch {
      // Silent fallback
    }
  }, []);

  const connect = useCallback(async () => {
    if (!enabled) return;

    try {
      const wsUrl = apiUrl.replace(/^http/, "ws") + "/api/v1/agent/voice";
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          switch (data.type) {
            case "status":
              setStatus(data.content);
              break;
            case "transcript":
              setTranscript(data.content);
              break;
            case "response_text":
              setResponseText((prev) => prev + data.content);
              break;
            case "response_audio":
              playAudioChunk(data.data);
              break;
            case "error":
              console.error("Voice error:", data.content);
              setStatus("error");
              break;
          }
        } catch {
          // skip
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsListening(false);
        setStatus("disconnected");
      };

      ws.onerror = () => {
        setStatus("error");
      };
    } catch (err) {
      console.error("Voice connection failed:", err);
    }
  }, [enabled, apiUrl, playAudioChunk]);

  const disconnect = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    wsRef.current?.close();
    wsRef.current = null;
    setIsConnected(false);
    setIsListening(false);
    setStatus("idle");
  }, []);

  const toggleListening = useCallback(async () => {
    if (!isConnected || !wsRef.current) return;

    if (isListening) {
      // Stop recording
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      wsRef.current.send(JSON.stringify({ type: "stop" }));
      setIsListening(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          const buffer = await event.data.arrayBuffer();
          const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
          wsRef.current.send(JSON.stringify({ type: "audio", data: base64 }));
        }
      };

      mediaRecorder.start(250); // Send chunks every 250ms
      setIsListening(true);
      setTranscript("");
      setResponseText("");
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }, [isConnected, isListening]);

  // Cleanup on unmount
  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  if (!enabled) return null;

  return (
    <div className="flex items-center gap-2">
      {/* Connect/Disconnect button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={isConnected ? disconnect : connect}
        className={cn(
          "p-2 rounded-lg transition-colors",
          isConnected
            ? "bg-red-500/10 text-red-500 hover:bg-red-500/20"
            : "bg-secondary text-muted-foreground hover:text-foreground"
        )}
        title={isConnected ? "Disconnect voice" : "Connect voice"}
      >
        {isConnected ? <PhoneOff size={16} /> : <Phone size={16} />}
      </motion.button>

      {/* Microphone button (only when connected) */}
      <AnimatePresence>
        {isConnected && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleListening}
            className={cn(
              "relative p-2 rounded-lg transition-colors",
              isListening
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-muted-foreground hover:text-foreground"
            )}
            title={isListening ? "Stop recording" : "Start recording"}
          >
            {isListening ? <MicOff size={16} /> : <Mic size={16} />}

            {/* Pulse animation when listening */}
            {isListening && (
              <motion.div
                className="absolute inset-0 rounded-lg border-2 border-primary"
                animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            )}
          </motion.button>
        )}
      </AnimatePresence>

      {/* Status indicator */}
      {isConnected && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-1 text-[10px] text-muted-foreground"
        >
          <div
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              status === "listening" && "bg-green-400",
              status === "processing" && "bg-amber-400 animate-pulse",
              status === "speaking" && "bg-blue-400",
              status === "connected" && "bg-green-400",
              status === "error" && "bg-red-400",
            )}
          />
          <span className="capitalize">{status}</span>
          {isListening && <Volume2 size={10} className="animate-pulse" />}
        </motion.div>
      )}
    </div>
  );
}
