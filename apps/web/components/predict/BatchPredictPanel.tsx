"use client";

import { useState } from "react";
import { UploadCloud, Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export function BatchPredictPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handlePredict = async () => {
    if (!file) return;
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/v1/predict/batch", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Batch prediction failed.");
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `predicted_${file.name}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error(e);
      alert("Failed to process batch CSV.");
    }
    
    setIsProcessing(false);
  };

  return (
    <div className="p-4 space-y-4">
      <div className="text-sm text-muted-foreground">
        Upload a CSV file containing chemical formulas (must include a <code className="text-primary font-mono bg-primary/10 px-1 rounded">formula</code> column).
      </div>
      
      <div className="border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center bg-muted/10 text-center hover:bg-muted/30 transition-colors cursor-pointer" onClick={() => document.getElementById('batch-csv-upload')?.click()}>
        <input 
          id="batch-csv-upload" 
          type="file" 
          accept=".csv" 
          className="hidden" 
          onChange={(e) => setFile(e.target.files?.[0] || null)} 
        />
        <UploadCloud className="w-8 h-8 text-emerald-500 mb-3" />
        <h3 className="font-semibold text-sm mb-1">{file ? file.name : "Select CSV Dataset"}</h3>
        <p className="text-xs text-muted-foreground">{file ? `${(file.size / 1024).toFixed(1)} KB` : "Max 1000 rows supported"}</p>
      </div>

      <Button onClick={handlePredict} disabled={!file || isProcessing} className="w-full gap-2" variant="outline">
        {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
        {isProcessing ? "Processing Batch..." : "Run Batch Prediction"}
      </Button>
    </div>
  );
}
