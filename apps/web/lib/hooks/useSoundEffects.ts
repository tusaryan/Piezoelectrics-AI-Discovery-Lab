/**
 * Sound effects hook for the Piezo.AI Agent chat.
 * Uses Web Audio API to generate procedural tones — no external audio files needed.
 * Sounds are toggleable and the preference is persisted to localStorage.
 */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type SoundType = "send" | "receive" | "toolCall" | "thinking";

export function useSoundEffects() {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const [enabled, setEnabled] = useState(true);

  // Load preference from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("piezo-sound-enabled");
    if (stored !== null) {
      setEnabled(stored === "true");
    }
  }, []);

  const toggle = useCallback(() => {
    setEnabled((prev) => {
      const next = !prev;
      localStorage.setItem("piezo-sound-enabled", String(next));
      return next;
    });
  }, []);

  const getCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }, []);

  const playSound = useCallback(
    (type: SoundType) => {
      if (!enabled) return;

      try {
        const ctx = getCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.connect(gain);
        gain.connect(ctx.destination);

        const now = ctx.currentTime;

        switch (type) {
          case "send":
            // Ascending chime: two quick notes
            osc.type = "sine";
            osc.frequency.setValueAtTime(523.25, now); // C5
            osc.frequency.setValueAtTime(659.25, now + 0.08); // E5
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
            osc.start(now);
            osc.stop(now + 0.2);
            break;

          case "receive":
            // Gentle notification: descending soft tone
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, now); // A5
            osc.frequency.exponentialRampToValueAtTime(659.25, now + 0.15);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
            osc.start(now);
            osc.stop(now + 0.25);
            break;

          case "toolCall":
            // Subtle click: very short burst
            osc.type = "triangle";
            osc.frequency.setValueAtTime(1200, now);
            osc.frequency.exponentialRampToValueAtTime(800, now + 0.03);
            gain.gain.setValueAtTime(0.06, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
            osc.start(now);
            osc.stop(now + 0.06);
            break;

          case "thinking":
            // Soft pulsing hum
            osc.type = "sine";
            osc.frequency.setValueAtTime(440, now);
            gain.gain.setValueAtTime(0.04, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
            osc.start(now);
            osc.stop(now + 0.15);
            break;
        }
      } catch {
        // Silently ignore AudioContext errors (e.g., before user interaction)
      }
    },
    [enabled, getCtx]
  );

  return { playSound, enabled, toggle };
}
