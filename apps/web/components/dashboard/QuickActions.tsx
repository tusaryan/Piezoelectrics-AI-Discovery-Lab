"use client";

import { useRouter } from "next/navigation";
import {
  BrainCircuit,
  Zap,
  FlaskConical,
  Eye,
  Database,
  Settings,
  ArrowRight,
} from "lucide-react";

const ACTIONS = [
  {
    label: "Train Model",
    description: "Configure and run ML training pipelines",
    icon: BrainCircuit,
    href: "/train",
    color: "#10B981",
  },
  {
    label: "Predict",
    description: "Run predictions on materials",
    icon: Zap,
    href: "/predict",
    color: "#F59E0B",
  },
  {
    label: "Upload Dataset",
    description: "Upload and process CSV datasets",
    icon: Database,
    href: "/dataset",
    color: "#4F46E5",
  },
  {
    label: "Optimization Lab",
    description: "Multi-objective property optimization",
    icon: FlaskConical,
    href: "/optimization-lab",
    color: "#EC4899",
  },
  {
    label: "Interpretability",
    description: "SHAP analysis & symbolic regression",
    icon: Eye,
    href: "/interpret",
    color: "#8B5CF6",
  },
  {
    label: "Settings",
    description: "System configuration & environment",
    icon: Settings,
    href: "/settings",
    color: "#64748B",
  },
];

export default function QuickActions() {
  const router = useRouter();

  return (
    <div className="quick-actions-section">
      <h2 className="section-title">Quick Actions</h2>
      <div className="quick-actions-grid">
        {ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.label}
              className="quick-action-btn"
              onClick={() => router.push(action.href)}
            >
              <div
                className="quick-action-icon"
                style={{ color: action.color }}
              >
                <Icon size={20} />
              </div>
              <div className="quick-action-text">
                <span className="quick-action-label">{action.label}</span>
                <span className="quick-action-desc">{action.description}</span>
              </div>
              <ArrowRight size={16} className="quick-action-arrow" />
            </button>
          );
        })}
      </div>
    </div>
  );
}
