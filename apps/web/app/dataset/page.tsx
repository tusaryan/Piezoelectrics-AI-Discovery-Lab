"use client";

import { useState, useEffect } from "react";
import { DatasetUploadZone } from "@/components/dataset/DatasetUploadZone";
import { SchemaMapper } from "@/components/dataset/SchemaMapper";
import { DataQualityReport } from "@/components/dataset/DataQualityReport";
import { MaterialsTable } from "@/components/dataset/MaterialsTable";
import { ArrowLeft, RotateCcw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSearchParams } from "next/navigation";

type Step = "upload" | "map" | "resolve" | "processing" | "table";

interface ColumnSuggestion {
  csv_column: string | null;
  confidence: number;
}

interface UploadResult {
  dataset_id: string;
  csv_columns: string[];
  row_count: number;
  suggested_mapping: Record<string, ColumnSuggestion>;
}

interface DataIssue {
  row_idx: number;
  column: string;
  issue_type: string;
  severity: "critical" | "warning" | "info";
  description: string;
  choices: string[];
}

interface MaterialData {
  id: string;
  dataset_id: string;
  formula: string;
  sintering_temp: number | null;
  d33: number | null;
  tc: number | null;
  is_imputed: boolean;
  is_tc_ai_generated: boolean;
}

const STEP_LABELS: Record<Step, string> = {
  upload: "Upload",
  map: "Map Columns",
  resolve: "Review Issues",
  processing: "Processing AI Features",
  table: "Dataset Ready",
};

const STEPS: Step[] = ["upload", "map", "resolve", "processing", "table"];

export default function DatasetPage() {
  const [step, setStep] = useState<Step>("upload");
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [issues, setIssues] = useState<DataIssue[]>([]);
  const [materials, setMaterials] = useState<MaterialData[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isResolving, setIsResolving] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isInitialized, setIsInitialized] = useState(false);
  
  const searchParams = useSearchParams();

  // Load from sessionStorage on mount or override via search params
  useEffect(() => {
    try {
      const pid = searchParams.get("id");
      const ptab = searchParams.get("tab");
      
      if (pid && ptab === "table") {
        setDatasetId(pid);
        setStep("table");
        fetchMaterials(pid);
        setIsInitialized(true);
        return;
      }
      
      const stored = sessionStorage.getItem("datasetState");
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.step) setStep(parsed.step);
        if (parsed.datasetId) setDatasetId(parsed.datasetId);
        if (parsed.uploadResult) setUploadResult(parsed.uploadResult);
        if (parsed.issues) setIssues(parsed.issues);
      }
    } catch (e) {
      console.error("Failed to load dataset state:", e);
    } finally {
      setIsInitialized(true);
    }
  }, [searchParams]);

  // Save to sessionStorage on changes
  useEffect(() => {
    if (!isInitialized) return; // don't overwrite with initial state before load
    
    // Only save if we are deeply in a flow, else clear it if upload step
    if (step === "upload") {
      sessionStorage.removeItem("datasetState");
    } else {
      sessionStorage.setItem("datasetState", JSON.stringify({
        step,
        datasetId,
        uploadResult,
        issues
      }));
    }
  }, [isInitialized, step, datasetId, uploadResult, issues]);

  // Progress is now controlled by SSE directly

  const resetToUpload = () => {
    sessionStorage.removeItem("datasetState");
    setStep("upload");
    setUploadResult(null);
    setIssues([]);
    setMaterials([]);
    setDatasetId(null);
  };

  const fetchMaterials = async (id: string) => {
    try {
      const res = await fetch(`/api/v1/datasets/${id}/materials?per_page=500`);
      if (!res.ok) return;
      const data = await res.json();
      setMaterials(data.data);
      setStep("table");
    } catch (e) {
      console.error(e);
    }
  };

  // Auto-fetch if user returns to page during table step but state memory has no materials
  useEffect(() => {
    if (isInitialized && step === "table" && materials.length === 0 && datasetId) {
      fetchMaterials(datasetId);
    }
  }, [isInitialized, step, materials.length, datasetId]);

  const startProgressStream = (id: string) => {
    // Connect directly to FastAPI — Next.js rewrites proxy buffers SSE streams
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const sse = new EventSource(`${apiBase}/api/v1/datasets/${id}/progress`);
    
    sse.onmessage = (e) => {
      const data = e.data;
      if (data === "100") {
        sse.close();
        fetchMaterials(id);
      } else if (data.startsWith("Error")) {
        sse.close();
        alert("Dataset processing failed on the backend:\n" + data);
        resetToUpload();
      } else {
        const parsed = parseInt(data, 10);
        if (!isNaN(parsed)) setProgress(parsed);
      }
    };
    
    sse.onerror = () => {
      sse.close();
      fetchMaterials(id); // fallback
    };
  };

  // Step 1: Upload file → get column suggestions
  const handleUpload = async (file: File) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/v1/datasets/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      const result: UploadResult = data.data;
      setUploadResult(result);
      setDatasetId(result.dataset_id);
      setStep("map");
    } catch (e: unknown) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to upload dataset.");
    } finally {
      setIsUploading(false);
    }
  };

  // Step 2: Confirm column mapping → validate
  const handleConfirmMapping = async (mapping: Record<string, string>) => {
    if (!datasetId) return;
    setIsUploading(true);
    try {
      const res = await fetch(`/api/v1/datasets/${datasetId}/confirm-mapping`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mapping }),
      });
      let data;
      const text = await res.text();
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error(res.ok ? "Failed to parse response" : text || res.statusText);
      }
      
      if (!res.ok) throw new Error(data.detail || "Mapping failed");

      const d = data.data;
      if (d.issues && d.issues.length > 0) {
        setIssues(d.issues);
        setStep("resolve");
      } else if (d.status === "processing") {
        setStep("processing");
        setProgress(0);
        startProgressStream(d.id);
      } else {
        fetchMaterials(d.id);
      }
    } catch (e: unknown) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to confirm mapping.");
    } finally {
      setIsUploading(false);
    }
  };

  // Step 3: Resolve quality issues
  const handleResolveIssues = async (resolutions: Record<number, string>) => {
    if (!datasetId) return;
    setIsResolving(true);
    try {
      const res = await fetch(`/api/v1/datasets/${datasetId}/resolve-issue`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resolutions }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Resolution failed");

      if (data.data.issues && data.data.issues.length > 0) {
        setIssues(data.data.issues);
      } else if (data.data.status === "processing") {
        setStep("processing");
        setProgress(0);
        startProgressStream(datasetId);
      } else {
        fetchMaterials(datasetId);
      }
    } catch (e: unknown) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to resolve issues.");
    } finally {
      setIsResolving(false);
    }
  };

  const currentStepIdx = STEPS.indexOf(step);

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Dataset Management</h1>
          <p className="text-muted-foreground">
            Upload, map, clean, and finalize your piezoelectric datasets for ML training.
          </p>
        </div>
        {step !== "upload" && (
          <button
            onClick={resetToUpload}
            className="inline-flex items-center gap-2 rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Start Over
          </button>
        )}
      </div>

      {/* Progress bar with labels */}
      <div className="flex gap-2 items-center justify-center w-full max-w-3xl mx-auto">
        {STEPS.map((s, idx) => (
          <div key={s} className="flex-1 flex flex-col items-center gap-1">
            <div
              className={`h-2 w-full rounded-full transition-colors ${
                idx <= currentStepIdx ? "bg-primary" : "bg-primary/20"
              }`}
            />
            <span
              className={`text-xs font-medium ${
                idx <= currentStepIdx ? "text-primary" : "text-muted-foreground"
              }`}
            >
              {STEP_LABELS[s]}
            </span>
          </div>
        ))}
      </div>

      {/* Navigation: Back button (visible on map and resolve steps) */}
      {(step === "map" || step === "resolve") && (
        <div className="flex justify-start">
          <button
            onClick={() => {
              if (step === "map") resetToUpload();
              else if (step === "resolve") setStep("map");
            }}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            {step === "map" ? "Re-upload File" : "Back to Column Mapping"}
          </button>
        </div>
      )}

      {step === "upload" && (
        <DatasetUploadZone onUpload={handleUpload} isUploading={isUploading} />
      )}

      {step === "map" && uploadResult && (
        <SchemaMapper
          csvColumns={uploadResult.csv_columns}
          suggestedMapping={uploadResult.suggested_mapping}
          onConfirm={handleConfirmMapping}
        />
      )}

      {step === "resolve" && (
        <DataQualityReport
          issues={issues}
          onResolve={handleResolveIssues}
          isResolving={isResolving}
        />
      )}

      {step === "processing" && (
        <div className="flex flex-col items-center justify-center py-24 space-y-8 animate-in fade-in zoom-in duration-500 max-w-md mx-auto">
          <Loader2 className="h-16 w-16 animate-spin text-primary" />
          <div className="text-center space-y-2 w-full">
            <h2 className="text-2xl font-bold tracking-tight">Processing AI Features</h2>
            <p className="text-muted-foreground">
              Generating physics-based ML features via matminer.
            </p>
          </div>
          
          <div className="w-full space-y-2">
            <div className="flex justify-between text-sm font-medium">
              <span>
                {progress < 5 ? "Parsing chemical formulas & structures..." :
                 progress < 15 ? "Computing elemental electronegativity & radii..." :
                 progress < 90 ? "Generating Advanced Physics Descriptors (35-dim)..." :
                 "Saving computed materials to Database..."}
              </span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-2 w-full bg-primary/20 rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary transition-all duration-500 ease-out" 
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {step === "table" && (
        <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b pb-4 gap-4">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-green-500 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                Dataset Processing Complete
              </h2>
              <p className="text-muted-foreground mt-1">Successfully extracted physics descriptors and indexed {materials.length} materials.</p>
            </div>
            <div className="flex gap-3">
              <Button variant="outline" onClick={resetToUpload}>Upload Another</Button>
              <Button onClick={() => window.location.href = "/dashboard"}>View Dashboard</Button>
              <Button variant="default" onClick={() => window.location.href = "/train"}>Train Models</Button>
            </div>
          </div>
          <MaterialsTable materials={materials} />
        </div>
      )}
    </div>
  );
}
