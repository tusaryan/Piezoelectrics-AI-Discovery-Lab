import { AlertCircle } from "lucide-react";

export function BetaWarning() {
  return (
    <div className="bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 p-3 rounded-lg flex items-start gap-3 text-sm mb-6">
      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <p>
        <strong>Note:</strong> This section currently displays <span className="font-semibold">mock UI and hardcoded data</span> to demonstrate the intended user experience. The backend machine learning pipelines for this feature are scheduled for implementation in upcoming phases.
      </p>
    </div>
  );
}
