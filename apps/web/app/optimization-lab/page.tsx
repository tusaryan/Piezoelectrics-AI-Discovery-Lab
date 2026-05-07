import {
  FlaskConical,
  Atom,
  Orbit,
  Target,
  Waypoints,
  TrendingUp,
  Tags,
  Box,
} from "lucide-react";

const FEATURES = [
  {
    icon: Atom,
    title: "Crystal Structure Analysis",
    description: "Generate approximate 3D perovskite structures from formulas with rapid structural relaxation",
  },
  {
    icon: Orbit,
    title: "Structural Feature Extraction",
    description: "Tolerance factor, octahedral factor, bond valence, Goldschmidt criteria — physics-based structural descriptors",
  },
  {
    icon: Box,
    title: "Structure Visualization",
    description: "Interactive 3D crystal structure display with comparison across different compositions",
  },
  {
    icon: Target,
    title: "Optimization Config",
    description: "Set target property ranges/constraints for d33, tc, hardness with use-case preset profiles",
  },
  {
    icon: Waypoints,
    title: "NSGA-II Optimization",
    description: "Multi-objective optimization using trained ML models to evaluate millions of theoretical compositions",
  },
  {
    icon: TrendingUp,
    title: "Pareto Front",
    description: "2D/3D interactive chart showing optimal trade-off surface between d33, tc, and hardness",
  },
  {
    icon: Tags,
    title: "Use-Case Mapping",
    description: "Each Pareto-optimal composition tagged with best-fit industrial use-case — wearables, actuators, transducers",
  },
  {
    icon: FlaskConical,
    title: "Solution Table",
    description: "Ranked list of Pareto-optimal compositions with predicted properties and convergence tracking",
  },
];

export default function OptimizationLabPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <FlaskConical size={22} />
        </div>
        <div className="page-header-text">
          <h1>Optimization Lab</h1>
          <p>Crystal structure analysis + multi-objective property optimization</p>
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
                <span className="feature-card-badge">Session 8</span>
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
