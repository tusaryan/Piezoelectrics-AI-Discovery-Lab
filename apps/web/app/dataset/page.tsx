import {
  Database,
  Upload,
  Columns3,
  AlertTriangle,
  Table2,
  Search,
  GitCompare,
  FileCheck,
} from "lucide-react";

const FEATURES = [
  {
    icon: Upload,
    title: "CSV Upload",
    description: "Drag-and-drop CSV upload with file size indicator and progress bar",
  },
  {
    icon: Columns3,
    title: "Column Mapping",
    description: "Mandatory mapping of CSV columns to backend schema fields — formula, d33, tc, hardness, and more",
  },
  {
    icon: AlertTriangle,
    title: "Review Issues",
    description: "Auto-detect invalid characters, missing values, unsupported elements. Inline fix, skip, or delete with multi-select",
  },
  {
    icon: Table2,
    title: "Dataset Explorer",
    description: "Virtualized table with sort, search, full CRUD on rows/cells, uid column, and Save/Cancel buttons",
  },
  {
    icon: FileCheck,
    title: "Data Quality Report",
    description: "Breakdown of issues per column, count of valid/invalid rows, status badges",
  },
  {
    icon: GitCompare,
    title: "Parsed Dataset Comparison",
    description: "Side-by-side source vs parsed view with uid mapping, mismatch highlighting, and search",
  },
  {
    icon: Search,
    title: "Formula Validation",
    description: "Parse and validate chemical formulas against the Central Element Registry at upload time",
  },
  {
    icon: Database,
    title: "Multi-Dataset Support",
    description: "Upload multiple datasets, view and select any from the dashboard with status tracking",
  },
];

export default function DatasetPage() {
  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-icon">
          <Database size={22} />
        </div>
        <div className="page-header-text">
          <h1>Dataset Upload & Management</h1>
          <p>Upload, map, clean, review, and explore CSV datasets</p>
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
                <span className="feature-card-badge">Session 2</span>
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
