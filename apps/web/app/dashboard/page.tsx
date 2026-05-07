import {
  BarChart3,
  FileText,
  BrainCircuit,
  Zap,
  Download,
  RefreshCw,
  PieChart,
  ListChecks,
} from "lucide-react";

const FEATURES = [
  {
    icon: BarChart3,
    title: "Stats Overview",
    description: "Datasets uploaded, trained models count, predictions made, and training jobs at a glance",
  },
  {
    icon: ListChecks,
    title: "Dataset Management",
    description: "View all datasets with rows & columns, delete, or open in the Dataset Explorer with full CRUD",
  },
  {
    icon: Zap,
    title: "Quick Actions",
    description: "One-click navigation to Train, Predict, Optimization Lab, and Interpretability",
  },
  {
    icon: BrainCircuit,
    title: "Model Library",
    description: "All trained models with R², RMSE, algorithm, rename (UUID stable), and download parsed dataset",
  },
  {
    icon: PieChart,
    title: "Target Distribution",
    description: "Donut chart showing model distribution across d33, tc, and hardness targets",
  },
  {
    icon: FileText,
    title: "Report Generation",
    description: "Premium PDF reports with R²/RMSE curves, predicted vs actual graphs, SHAP analysis, and AI insights",
  },
  {
    icon: RefreshCw,
    title: "Default Model",
    description: "Select which trained model is used for predictions across the platform",
  },
  {
    icon: Download,
    title: "Data Export",
    description: "Download parsed datasets and trained model artifacts for manual verification",
  },
];

export default function DashboardPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <BarChart3 size={22} />
        </div>
        <div className="page-header-text">
          <h1>Dashboard</h1>
          <p>System overview, quick actions, report generation, model management</p>
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
                <span className="feature-card-badge">Session 6</span>
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
