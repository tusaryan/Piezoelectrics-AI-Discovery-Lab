import {
  Zap,
  FlaskConical,
  Gauge,
  FileSpreadsheet,
  GitCompareArrows,
  FileDown,
  Atom,
  Shield,
} from "lucide-react";

const FEATURES = [
  {
    icon: Atom,
    title: "Formula Input",
    description: "Enter chemical compositions with auto-fix for unicode subscripts, parentheses, and decimal notation support",
  },
  {
    icon: FlaskConical,
    title: "Composite Fields",
    description: "Matrix type, filler wt%, particle morphology/size, surface treatment — shown only if model supports composites",
  },
  {
    icon: Gauge,
    title: "Prediction Output",
    description: "d33 (pC/N), tc (°C), Vickers Hardness (HV) with 95% confidence intervals and animated gauge bars",
  },
  {
    icon: Shield,
    title: "Mohs Scale",
    description: "Visual Mohs Hardness Scale representation when hardness prediction is available",
  },
  {
    icon: GitCompareArrows,
    title: "Multi-Material Comparison",
    description: "Add multiple predictions side-by-side, compare d33/tc/hardness across bulk ceramics and composites",
  },
  {
    icon: FileSpreadsheet,
    title: "Batch Processing",
    description: "Upload CSV → predict all rows → download results with d33_predicted, tc_predicted columns appended",
  },
  {
    icon: FileDown,
    title: "Report Download",
    description: "Select predictions to include in PDF report with optional AI insights about use-cases",
  },
  {
    icon: Zap,
    title: "Smart Detection",
    description: "Auto-detect bulk vs composite: filler_wt_pct=0 + matrix_type='none' → bulk ceramic; else composite",
  },
];

export default function PredictPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <Zap size={22} />
        </div>
        <div className="page-header-text">
          <h1>Predict</h1>
          <p>Unified prediction: bulk ceramics + composites + hardness</p>
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
                <span className="feature-card-badge">Session 5</span>
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
