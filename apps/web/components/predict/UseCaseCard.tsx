"use client";

/**
 * UseCaseCard — Display predicted industrial use-cases with confidence tiers,
 * driving properties, multiple recommendations, and scientific caution notes.
 */

import { Target, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { usePredictStore } from "@/lib/store/predictStore";

interface UsageRecommendation {
  name: string;
  score: number;
  confidence: number;
  tier: string;
  tier_label: string;
  description: string;
  icon: string;
  color: string;
  driving_properties: string[];
}

interface UsagePredictions {
  recommendations: UsageRecommendation[];
  caution_notes: string[];
  property_completeness: string;
  properties_used: string[];
}

export default function UseCaseCard() {
  const { prediction } = usePredictStore();
  const [showMore, setShowMore] = useState(false);

  if (!prediction || prediction.status !== "success" || !prediction.use_case) {
    return null;
  }

  const uc = prediction.use_case;
  const usage = prediction.usage_predictions as UsagePredictions | null;
  const extraRecs = usage?.recommendations?.slice(1, 4) || [];
  const cautions = usage?.caution_notes || [];

  // Score display: prefer score field, fallback to confidence * 100
  const getScore = (item: any): number => {
    if (item.score != null) return item.score;
    if (item.confidence != null) return Math.round(item.confidence * 100);
    return 0;
  };

  const tierBadge = (tier: string, label: string, color: string) => {
    const bgMap: Record<string, string> = {
      primary: "#10B98120",
      secondary: "#F59E0B20",
      tertiary: "#EF444420",
    };
    const textMap: Record<string, string> = {
      primary: "#10B981",
      secondary: "#F59E0B",
      tertiary: "#EF4444",
    };
    return (
      <span
        className="use-case-tier-badge"
        style={{
          background: bgMap[tier] || `${color}15`,
          color: textMap[tier] || color,
          fontSize: 9,
          fontWeight: 700,
          padding: "2px 6px",
          borderRadius: 4,
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          marginLeft: 6,
        }}
      >
        {label}
      </span>
    );
  };

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <Target size={16} /> Suggested Use Cases
        {usage?.property_completeness === "partial" && (
          <span style={{ fontSize: 10, color: "var(--text-muted)", fontWeight: 400, marginLeft: 8 }}>
            (Estimated — {usage.properties_used.length}/3 properties available)
          </span>
        )}
      </div>

      {/* Primary recommendation */}
      <div className="use-case-card" style={{ borderLeft: `3px solid ${uc.color}` }}>
        <div
          className="use-case-icon"
          style={{ background: `${uc.color}15`, fontSize: 20 }}
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
              {getScore(uc)}%
            </span>
            {uc.tier_label && tierBadge(uc.tier || 'secondary', uc.tier_label, uc.color)}
          </div>
          <p className="use-case-description">{uc.description}</p>
          {uc.driving_properties && uc.driving_properties.length > 0 && (
            <div className="use-case-drivers" style={{
              marginTop: 4, fontSize: 10, color: "var(--text-muted)",
              display: "flex", gap: 6, flexWrap: "wrap",
            }}>
              {uc.driving_properties.map((prop: string, i: number) => (
                <span key={i} style={{
                  background: "var(--surface-hover)",
                  padding: "1px 6px",
                  borderRadius: 3,
                  fontFamily: "monospace",
                }}>
                  {prop}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Additional recommendations (collapsible) */}
      {extraRecs.length > 0 && (
        <>
          <button
            onClick={() => setShowMore(!showMore)}
            style={{
              display: "flex", alignItems: "center", gap: 4,
              background: "none", border: "none", color: "var(--text-muted)",
              cursor: "pointer", fontSize: 11, padding: "6px 0", width: "100%",
            }}
          >
            {showMore ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {showMore ? "Hide" : "Show"} {extraRecs.length} more suggestion{extraRecs.length > 1 ? "s" : ""}
          </button>

          {showMore && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {extraRecs.map((rec) => (
                <div
                  key={rec.name}
                  className="use-case-card"
                  style={{
                    padding: "8px 10px",
                    borderLeft: `2px solid ${rec.color}`,
                    opacity: 0.85,
                  }}
                >
                  <div className="use-case-icon" style={{ background: `${rec.color}10`, fontSize: 16, width: 28, height: 28 }}>
                    {rec.icon}
                  </div>
                  <div className="use-case-content">
                    <div className="use-case-name" style={{ fontSize: 12 }}>
                      {rec.name}
                      <span
                        className="use-case-confidence"
                        style={{
                          background: `${rec.color}15`,
                          color: rec.color,
                          fontSize: 10,
                          padding: "1px 6px",
                        }}
                      >
                        {getScore(rec)}%
                      </span>
                      {rec.tier_label && tierBadge(rec.tier || 'tertiary', rec.tier_label, rec.color)}
                    </div>
                    <p className="use-case-description" style={{ fontSize: 10, marginTop: 2 }}>
                      {rec.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Scientific caution notes */}
      {cautions.length > 0 && (
        <div style={{
          marginTop: 8, display: "flex", flexDirection: "column", gap: 4,
        }}>
          {cautions.map((note, i) => (
            <div
              key={i}
              style={{
                display: "flex", alignItems: "flex-start", gap: 6,
                padding: "6px 10px",
                background: "var(--warning-bg, rgba(245, 158, 11, 0.08))",
                borderRadius: 6,
                borderLeft: "2px solid var(--warning-color, #F59E0B)",
                fontSize: 11,
                color: "var(--text-secondary)",
                lineHeight: 1.4,
              }}
            >
              <AlertTriangle size={12} style={{ flexShrink: 0, marginTop: 1, color: "var(--warning-color, #F59E0B)" }} />
              <span>{note}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
