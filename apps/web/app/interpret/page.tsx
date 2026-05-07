import {
  Eye,
  ScatterChart,
  Waves,
  TrendingDown,
  ShieldCheck,
  Sigma,
  Maximize2,
  Info,
} from "lucide-react";

const FEATURES = [
  {
    icon: ScatterChart,
    title: "SHAP Beeswarm Plot",
    description: "Global feature importance — which features most impact predictions, color-coded by feature value",
  },
  {
    icon: Waves,
    title: "SHAP Waterfall Plot",
    description: "Local prediction explanation showing contribution of each feature to a specific single prediction",
  },
  {
    icon: TrendingDown,
    title: "Feature Dependence Plot",
    description: "Relationship between a specific feature and its SHAP values, revealing non-linear relationships",
  },
  {
    icon: ShieldCheck,
    title: "Physics Validation",
    description: "Checks if SHAP associations align with expected solid-state physics — alignment score and violations",
  },
  {
    icon: Sigma,
    title: "Symbolic Regression (PySR)",
    description: "Discover interpretable mathematical equations relating composition features to properties with KaTeX rendering",
  },
  {
    icon: TrendingDown,
    title: "Parsimony Pressure",
    description: "Accuracy vs complexity trade-off — Pareto front of equations showing optimal simplicity-accuracy balance",
  },
  {
    icon: Maximize2,
    title: "Expandable Graphs",
    description: "Full-window view with zoom, pan, move controls. Hide controls button for clean screenshots",
  },
  {
    icon: Info,
    title: "Info Tooltips",
    description: "Every plot has tooltips explaining what it represents, its significance, and how to interpret results",
  },
];

export default function InterpretPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <Eye size={22} />
        </div>
        <div className="page-header-text">
          <h1>Interpretability</h1>
          <p>SHAP analysis, physics validation, and symbolic regression (PySR)</p>
        </div>
      </div>

      <div className="feature-grid">
        {FEATURES.map((feature) => {
          const Icon = feature.icon;
          return (
            <div key={feature.title} className="feature-card">
              <div className="feature-card-header">
                <div className="feature-card-icon">
                  <Icon size={18} />
                </div>
                <span className="feature-card-badge">Session 7</span>
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
