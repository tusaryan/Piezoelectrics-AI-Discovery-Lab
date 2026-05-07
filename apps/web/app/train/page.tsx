import {
  BrainCircuit,
  Sliders,
  Terminal,
  TrendingUp,
  Square,
  BarChart2,
  Layers,
  Timer,
} from "lucide-react";

const FEATURES = [
  {
    icon: Sliders,
    title: "Pipeline Configurator",
    description: "Select dataset, target fields, algorithms, and fine-tune hyperparameters with constrained ranges and \"i\" tooltips",
  },
  {
    icon: Layers,
    title: "Algorithm Selection",
    description: "XGBoost, Random Forest, SVM/SVR, LightGBM, Gradient Boosting, Decision Tree, ANN, Stacking — per-target or unified",
  },
  {
    icon: Terminal,
    title: "Real-Time Terminal",
    description: "Live backend log streaming: preprocessing steps, train/test split, feature engineering, training progress with color coding",
  },
  {
    icon: Timer,
    title: "Progress Bar",
    description: "Real-time completion percentage advancing per ML stage with proportional weights and current stage label",
  },
  {
    icon: TrendingUp,
    title: "Convergence Chart",
    description: "Real-time plotting of actual model convergence metric with info guide showing good vs bad patterns",
  },
  {
    icon: Square,
    title: "Stop Button",
    description: "Abort training mid-process — signal propagated to backend ML process. State: hidden → play → stop → checkmark",
  },
  {
    icon: BarChart2,
    title: "Results & Comparison",
    description: "Predicted vs Actual scatter plots, R²/RMSE metrics, cross-model comparison bar charts",
  },
  {
    icon: BrainCircuit,
    title: "Auto-Tune (Optuna)",
    description: "Toggle between Manual (user sets params) and Auto (Optuna-tuned) training modes for optimal hyperparameters",
  },
];

export default function TrainPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <BrainCircuit size={22} />
        </div>
        <div className="page-header-text">
          <h1>Model Studio</h1>
          <p>Configure ML pipelines, train models, view convergence and results</p>
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
                <span className="feature-card-badge">Session 4</span>
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
