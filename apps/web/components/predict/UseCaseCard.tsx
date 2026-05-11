"use client";

/**
 * UseCaseCard — Display predicted industrial use-case.
 */

import { Target } from "lucide-react";
import { usePredictStore } from "@/lib/store/predictStore";

export default function UseCaseCard() {
  const { prediction } = usePredictStore();

  if (!prediction || prediction.status !== "success" || !prediction.use_case) {
    return null;
  }

  const uc = prediction.use_case;

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <Target size={16} /> Suggested Use Case
      </div>
      <div className="use-case-card">
        <div
          className="use-case-icon"
          style={{ background: `${uc.color}15` }}
        >
          {uc.icon}
        </div>
        <div className="use-case-content">
          <div className="use-case-name">
            {uc.name}
            <span
              className="use-case-confidence"
              style={{
                background: `${uc.color}15`,
                color: uc.color,
              }}
            >
              {(uc.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <p className="use-case-description">{uc.description}</p>
        </div>
      </div>
    </div>
  );
}
