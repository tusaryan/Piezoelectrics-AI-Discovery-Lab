"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, FileType, CheckCircle, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DatasetUploadZoneProps {
  onUpload: (file: File) => void;
  isUploading?: boolean;
}

export function DatasetUploadZone({ onUpload, isUploading }: DatasetUploadZoneProps) {
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null);
    if (rejectedFiles.length > 0) {
      setError("Please upload a valid CSV or Excel (.xlsx) file under 50MB.");
      return;
    }
    
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: false,
    disabled: isUploading
  });

  return (
    <div className="w-full max-w-2xl mx-auto mt-8">
      <div 
        {...getRootProps()} 
        className={cn(
          "relative group p-12 rounded-xl border-2 border-dashed transition-all duration-300 ease-out flex flex-col items-center justify-center cursor-pointer overflow-hidden bg-background/50",
          isDragActive 
            ? "border-primary bg-primary/5 scale-[1.02]" 
            : "border-muted-foreground/30 hover:border-primary/50 hover:bg-muted/30",
          isUploading ? "opacity-50 cursor-not-allowed" : ""
        )}
      >
        <input {...getInputProps()} />
        
        {/* Animated Background Gradient for Quantum Indigo feel */}
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/0 via-purple-500/0 to-cyan-500/0 opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
        
        <motion.div
          animate={isDragActive ? { y: -10, scale: 1.1 } : { y: 0, scale: 1 }}
          className="relative z-10 flex text-primary mb-4 justify-center items-center"
        >
          {isUploading ? (
            <div className="h-16 w-16 rounded-full border-4 border-primary/30 border-t-primary animate-spin" />
          ) : (
            <UploadCloud className={cn("w-16 h-16", isDragActive ? "text-primary" : "text-muted-foreground")} />
          )}
        </motion.div>
        
        <div className="relative z-10 text-center space-y-2">
          <h3 className="text-xl font-semibold text-foreground">
            {isUploading ? "Uploading Dataset..." : "Drag & Drop your dataset"}
          </h3>
          <p className="text-sm text-muted-foreground">
            {isDragActive ? "Drop the file to start" : "or click to browse your files"}
          </p>
          <div className="flex items-center justify-center gap-2 mt-4 text-xs font-mono text-muted-foreground/80">
            <span className="flex items-center"><FileType className="w-3 h-3 mr-1"/> CSV</span>
            <span>•</span>
            <span className="flex items-center"><FileType className="w-3 h-3 mr-1"/> XLSX</span>
            <span>•</span>
            <span>Max 50MB</span>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-4 p-4 rounded-lg bg-destructive/10 border border-destructive/20 flex items-center text-destructive text-sm"
          >
            <AlertCircle className="w-4 h-4 mr-2 shrink-0" />
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
