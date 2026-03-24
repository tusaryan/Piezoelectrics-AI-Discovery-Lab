"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, XCircle, Check, Pencil, X, Wand2, Info } from "lucide-react";

interface DataIssue {
  row_idx: number;
  column: string;
  issue_type: string;
  severity: "critical" | "warning" | "info";
  description: string;
  choices: string[];
  auto_fixable?: boolean;
}

interface DataQualityReportProps {
  issues: DataIssue[];
  onResolve: (resolutions: Record<number, string>) => void;
  isResolving?: boolean;
}

export function DataQualityReport({ issues, onResolve, isResolving }: DataQualityReportProps) {
  const [resolutions, setResolutions] = useState<Record<string, string>>({});
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [editingKey, setEditingKey] = useState<string | null>(null);

  // Use composite key (row_idx + column + issue_type) for unique identification
  const issueKey = (issue: DataIssue) => `${issue.row_idx}:${issue.column}:${issue.issue_type}`;

  const autoFixableIssues = issues.filter(
    (i) => i.choices.includes("Auto-Fix") || i.choices.includes("Proceed Anyway")
  );
  const criticals = issues.filter((i) => i.severity === "critical");
  const warnings = issues.filter((i) => i.severity === "warning");
  const infos = issues.filter((i) => i.severity === "info");

  // Sort: unresolved first, then by severity (critical > warning > info)
  const sortedIssues = useMemo(() => {
    const sevOrder: Record<string, number> = { critical: 0, warning: 1, info: 2 };
    return [...issues].sort((a, b) => {
      const aResolved = !!resolutions[issueKey(a)];
      const bResolved = !!resolutions[issueKey(b)];
      if (aResolved !== bResolved) return aResolved ? 1 : -1;
      return (sevOrder[a.severity] ?? 3) - (sevOrder[b.severity] ?? 3);
    });
  }, [issues, resolutions]);

  if (issues.length === 0) return null;

  // All critical issues must have a resolution
  const allCriticalsResolved = criticals.every((c) => resolutions[issueKey(c)]);
  // For finalization: all issues should have a resolution or be info-level
  const canFinalize = allCriticalsResolved;

  const handleChoice = (key: string, choice: string) => {
    if (choice === "Edit Manually") {
      setEditingKey(key);
      // Pre-fill with the invalid value which is often embedded in the description:
      // "Cannot parse formula '...'" or "d33 = 280 is outside..."
      // We can grab the broken string out of the description via regex
      const issueDetails = issues.find(i => issueKey(i) === key);
      let prefill = "";
      if (issueDetails) {
        const match = issueDetails.description.match(/'([^']+)'/);
        if (match) prefill = match[1];
        else {
          const eqMatch = issueDetails.description.match(/=\s*([^ ]+)/);
          if (eqMatch) prefill = eqMatch[1];
        }
      }
      setEditValues((prev) => ({ ...prev, [key]: prefill }));
      return;
    }
    setResolutions((prev) => ({ ...prev, [key]: choice }));
  };

  const handleEditConfirm = async (key: string) => {
    const value = editValues[key];
    if (value && value.trim()) {
      // Optional check to ensure it at least doesn't contain weird characters
      // (Backend will ultimately validate it during resolution)
      setResolutions((prev) => ({ ...prev, [key]: `edit:${value.trim()}` }));
      setEditingKey(null);
    }
  };

  const handleAutoFixAll = () => {
    const newResolutions = { ...resolutions };
    for (const issue of issues) {
      const key = issueKey(issue);
      if (newResolutions[key]) continue; // Don't override existing choices
      if (issue.choices.includes("Auto-Fix")) {
        newResolutions[key] = "Auto-Fix";
      } else if (issue.choices.includes("Proceed Anyway")) {
        newResolutions[key] = "Proceed Anyway";
      } else if (issue.choices.includes("Keep Empty")) {
        newResolutions[key] = "Keep Empty";
      }
    }
    setResolutions(newResolutions);
  };
  
  const handleDropCriticals = () => {
    const newResolutions = { ...resolutions };
    for (const issue of criticals) {
      const key = issueKey(issue);
      if (newResolutions[key]) continue;
      if (issue.choices.includes("Drop Row")) {
        newResolutions[key] = "Drop Row";
      }
    }
    setResolutions(newResolutions);
  };
  
  const handleKeepSuspicious = () => {
    const newResolutions = { ...resolutions };
    for (const issue of issues) {
      const key = issueKey(issue);
      if (newResolutions[key]) continue;
      if (issue.choices.includes("Keep (suspicious)")) {
        newResolutions[key] = "Keep (suspicious)";
      }
    }
    setResolutions(newResolutions);
  };

  // Transmit exact composite key dictionary back to API, not truncating to row_index
  const handleFinalize = () => {
    onResolve(resolutions);
  };

  const resolvedCount = Object.keys(resolutions).length;
  const sevIcon = (sev: string) => {
    if (sev === "critical") return <XCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />;
    if (sev === "warning") return <AlertTriangle className="w-5 h-5 text-yellow-600 shrink-0 mt-0.5" />;
    return <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />;
  };

  return (
    <div className="w-full max-w-5xl mx-auto rounded-xl border bg-card p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xl font-semibold mb-1">Data Quality Report</h3>
          <p className="text-sm text-muted-foreground">
            Review and resolve issues, then finalize the dataset.
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {criticals.length > 0 && (
            <span className="inline-flex items-center rounded-full bg-destructive/10 px-2.5 py-0.5 text-xs font-semibold text-destructive">
              {criticals.length} Critical
            </span>
          )}
          {warnings.length > 0 && (
            <span className="inline-flex items-center rounded-full bg-yellow-500/10 px-2.5 py-0.5 text-xs font-semibold text-yellow-600">
              {warnings.length} Warnings
            </span>
          )}
          {infos.length > 0 && (
            <span className="inline-flex items-center rounded-full bg-blue-500/10 px-2.5 py-0.5 text-xs font-semibold text-blue-600">
              {infos.length} Auto-fixable
            </span>
          )}
        </div>
      </div>

      {/* Bulk Action bars */}
      <div className="flex flex-col gap-2 mb-4">
        {autoFixableIssues.length > 0 && (
          <div className="p-3 rounded-lg bg-blue-500/5 border border-blue-500/20 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-blue-700 dark:text-blue-400">
              <Wand2 className="w-4 h-4" />
              <span>
                <strong>{autoFixableIssues.length}</strong> issues can be auto-fixed (Unicode normalization, missing values).
              </span>
            </div>
            <button
              onClick={handleAutoFixAll}
              className="inline-flex items-center gap-1.5 rounded-md bg-blue-600 text-white text-xs font-medium h-8 px-4 hover:bg-blue-700 transition-colors"
            >
              <Wand2 className="w-3.5 h-3.5" />
              Auto-Fix All
            </button>
          </div>
        )}
        
        {criticals.length > 0 && (
          <div className="p-3 rounded-lg bg-destructive/5 border border-destructive/20 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-destructive">
              <XCircle className="w-4 h-4" />
              <span>
                <strong>{criticals.length}</strong> critical issues preventing dataset completion.
              </span>
            </div>
            <button
              onClick={handleDropCriticals}
              className="inline-flex items-center gap-1.5 rounded-md bg-destructive text-white text-xs font-medium h-8 px-4 hover:bg-destructive/90 transition-colors"
            >
              Drop All Criticals
            </button>
          </div>
        )}
        
        {warnings.length > 0 && (
          <div className="p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-600">
              <AlertTriangle className="w-4 h-4" />
              <span>
                <strong>{warnings.length}</strong> suspicious or out-of-range values detected.
              </span>
            </div>
            <button
              onClick={handleKeepSuspicious}
              className="inline-flex items-center gap-1.5 rounded-md bg-yellow-600 text-white text-xs font-medium h-8 px-4 hover:bg-yellow-700 transition-colors"
            >
              Keep Suspicious Data
            </button>
          </div>
        )}
      </div>

      <div className="space-y-2 mb-6 max-h-[500px] overflow-y-auto pr-2">
        {sortedIssues.map((issue) => {
          const key = issueKey(issue);
          const selectedChoice = resolutions[key];
          const isEditing = editingKey === key;
          const isResolved = !!selectedChoice;

          return (
            <motion.div
              key={key}
              layout
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`p-3 rounded-lg border transition-colors ${
                isResolved
                  ? "border-muted bg-muted/30 opacity-70"
                  : issue.severity === "critical"
                  ? "border-destructive/30 bg-destructive/5"
                  : issue.severity === "warning"
                  ? "border-yellow-500/30 bg-yellow-500/5"
                  : "border-blue-500/20 bg-blue-500/5"
              }`}
            >
              <div className="flex items-start gap-3">
                {isResolved ? (
                  <Check className="w-5 h-5 text-emerald-500 shrink-0 mt-0.5" />
                ) : (
                  sevIcon(issue.severity)
                )}

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-sm">
                      {issue.row_idx >= 0 ? `Row ${issue.row_idx}` : "Global"}: {issue.column}
                    </span>
                    {isResolved && (
                      <span className="text-xs text-emerald-600 font-medium">
                        ✓ {selectedChoice?.startsWith("edit:") ? `Edited: ${selectedChoice.slice(5)}` : selectedChoice}
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground mb-2 break-all">
                    {issue.description}
                  </div>

                  {/* Inline edit UI */}
                  {isEditing && (
                    <div className="flex gap-2 items-center mb-2">
                      <input
                        type="text"
                        autoFocus
                        placeholder={`Enter corrected ${issue.column} value...`}
                        value={editValues[key] || ""}
                        onChange={(e) =>
                          setEditValues((prev) => ({ ...prev, [key]: e.target.value }))
                        }
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleEditConfirm(key);
                          if (e.key === "Escape") setEditingKey(null);
                        }}
                        className="flex-1 h-8 px-3 rounded-md border border-input bg-background text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-mono"
                      />
                      <button
                        onClick={() => handleEditConfirm(key)}
                        className="inline-flex items-center rounded-md bg-primary text-primary-foreground h-8 px-3 text-xs font-medium hover:bg-primary/90 transition-colors"
                      >
                        <Check className="w-3.5 h-3.5 mr-1" />
                        Apply
                      </button>
                      <button
                        onClick={() => setEditingKey(null)}
                        className="inline-flex items-center rounded-md border bg-background h-8 px-3 text-xs text-muted-foreground hover:bg-muted transition-colors"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}

                  {/* Choice buttons */}
                  {!isResolved && !isEditing && (
                    <div className="flex gap-1.5 flex-wrap">
                      {issue.choices.map((choice) => (
                        <button
                          key={choice}
                          onClick={() => handleChoice(key, choice)}
                          className="inline-flex items-center rounded-md text-xs font-medium transition-colors border h-7 px-3 bg-background hover:bg-muted text-muted-foreground"
                        >
                          {choice === "Edit Manually" && <Pencil className="w-3 h-3 mr-1" />}
                          {choice === "Auto-Fix" && <Wand2 className="w-3 h-3 mr-1" />}
                          {choice}
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Undo button for resolved issues */}
                  {isResolved && !isEditing && (
                    <button
                      onClick={() => {
                        setResolutions((prev) => {
                          const next = { ...prev };
                          delete next[key];
                          return next;
                        });
                      }}
                      className="text-xs text-muted-foreground hover:text-foreground underline"
                    >
                      Undo
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="flex justify-between items-center border-t pt-4">
        <div className="text-sm text-muted-foreground">
          {resolvedCount} of {issues.length} issues resolved
          {criticals.length > 0 && !allCriticalsResolved && (
            <span className="text-destructive ml-2">
              ({criticals.filter((c) => !resolutions[issueKey(c)]).length} critical remaining)
            </span>
          )}
        </div>
        <button
          disabled={!canFinalize || isResolving}
          onClick={handleFinalize}
          className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground text-sm font-medium h-10 px-6 disabled:opacity-50 disabled:pointer-events-none hover:bg-primary/90 transition-colors"
        >
          {isResolving ? (
            <div className="h-4 w-4 rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground animate-spin mr-2" />
          ) : (
            <Check className="w-4 h-4 mr-2" />
          )}
          Finalize Dataset
        </button>
      </div>
    </div>
  );
}
