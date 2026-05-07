import {
  Settings,
  Server,
  BrainCircuit,
  Atom,
  Key,
  AlertTriangle,
  Trash2,
  Globe,
} from "lucide-react";

const FEATURES = [
  {
    icon: Server,
    title: "System Environment",
    description: "Dataset count, trained models count, predictions count, database size overview",
  },
  {
    icon: BrainCircuit,
    title: "Trained Models Library",
    description: "Table with target, model name (renameable), algorithm, R², RMSE, created date — set default, rename, delete",
  },
  {
    icon: Atom,
    title: "Pending Elements",
    description: "Add new elements to the registry, edit supported fields, auto-bootstrap from mendeleev/pymatgen on startup",
  },
  {
    icon: BrainCircuit,
    title: "Default Model Selector",
    description: "Choose which trained model is used for predictions globally across the platform",
  },
  {
    icon: Globe,
    title: "API Configuration",
    description: "Backend URL, WebSocket URL, database URL, LLM API key and model selection",
  },
  {
    icon: Key,
    title: "LLM Settings",
    description: "Configure OpenAI, Anthropic, Google, or local Ollama models with custom temperature and max tokens",
  },
  {
    icon: Trash2,
    title: "Purge All Models",
    description: "Delete all trained models from database and filesystem — requires confirmation",
  },
  {
    icon: AlertTriangle,
    title: "Clear Cache",
    description: "Remove cached prediction results and generated reports — requires confirmation",
  },
];

export default function SettingsPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <Settings size={22} />
        </div>
        <div className="page-header-text">
          <h1>Settings</h1>
          <p>Models library, system environment, API configuration, danger zone</p>
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
                <span className="feature-card-badge">Session 9</span>
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
