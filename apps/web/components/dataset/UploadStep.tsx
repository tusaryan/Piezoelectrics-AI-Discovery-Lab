"use client";

/**
 * Piezo.AI — Upload Step
 * ========================
 * Drag-and-drop CSV upload with file info card and progress bar.
 * Step 1 of the dataset upload wizard.
 */

import { useCallback, useState } from "react";
import { Upload, FileText, X, AlertCircle, CheckCircle2 } from "lucide-react";
import { useDatasetStore } from "@/lib/store/datasetStore";
import { uploadCSV } from "@/lib/api/datasets";

/* ---------- Helpers ---------- */

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ---------- Component ---------- */

export default function UploadStep() {
  const {
    uploadedFile,
    uploadProgress,
    isUploading,
    uploadError,
    setUploadedFile,
    setUploadProgress,
    setIsUploading,
    setUploadError,
    setUploadPreview,
    setWizardStep,
  } = useDatasetStore();

  const [isDragOver, setIsDragOver] = useState(false);

  /* Drag handlers */
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      const files = e.dataTransfer.files;
      if (files.length === 0) return;

      const file = files[0];
      if (!file.name.toLowerCase().endsWith(".csv")) {
        setUploadError("Only CSV files are accepted. Please upload a .csv file.");
        return;
      }
      setUploadedFile(file);
    },
    [setUploadedFile, setUploadError],
  );

  /* File input handler */
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".csv")) {
        setUploadError("Only CSV files are accepted. Please upload a .csv file.");
        return;
      }
      setUploadedFile(file);
    },
    [setUploadedFile, setUploadError],
  );

  /* Upload action */
  const handleUpload = useCallback(async () => {
    if (!uploadedFile) return;

    setIsUploading(true);
    setUploadError(null);
    setUploadProgress(0);

    try {
      const preview = await uploadCSV(uploadedFile, (pct) => {
        setUploadProgress(pct);
      });

      setUploadPreview(preview);
      setUploadProgress(100);

      // Brief delay to show 100% before advancing
      setTimeout(() => {
        setIsUploading(false);
        setWizardStep("map");
      }, 400);
    } catch (err) {
      setIsUploading(false);
      setUploadError(err instanceof Error ? err.message : "Upload failed. Please try again.");
    }
  }, [uploadedFile, setIsUploading, setUploadError, setUploadProgress, setUploadPreview, setWizardStep]);

  /* Clear file */
  const handleClear = useCallback(() => {
    setUploadedFile(null);
    setUploadError(null);
    setUploadProgress(0);
  }, [setUploadedFile, setUploadError, setUploadProgress]);

  return (
    <div className="upload-step">
      {/* Drop zone */}
      {!uploadedFile && (
        <div
          className={`upload-dropzone${isDragOver ? " active" : ""}${uploadError ? " rejected" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById("csv-file-input")?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload CSV file"
        >
          <input
            id="csv-file-input"
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />
          <div className="upload-dropzone-icon">
            <Upload size={32} />
          </div>
          <p className="upload-dropzone-title">
            Drag &amp; drop your CSV file here
          </p>
          <p className="upload-dropzone-subtitle">
            or <span className="upload-dropzone-link">click to browse</span>
          </p>
          <p className="upload-dropzone-hint">
            Supports .csv files with comma-separated values
          </p>
        </div>
      )}

      {/* Error message */}
      {uploadError && !uploadedFile && (
        <div className="upload-error">
          <AlertCircle size={16} />
          <span>{uploadError}</span>
        </div>
      )}

      {/* File info card */}
      {uploadedFile && (
        <div className="upload-file-info">
          <div className="upload-file-icon">
            <FileText size={24} />
          </div>
          <div className="upload-file-details">
            <p className="upload-file-name">{uploadedFile.name}</p>
            <p className="upload-file-size">{formatFileSize(uploadedFile.size)}</p>
          </div>
          {!isUploading && (
            <button
              className="upload-file-remove"
              onClick={handleClear}
              aria-label="Remove file"
            >
              <X size={16} />
            </button>
          )}
        </div>
      )}

      {/* Progress bar */}
      {isUploading && (
        <div className="upload-progress-container">
          <div className="upload-progress-bar">
            <div
              className="upload-progress-fill"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <div className="upload-progress-info">
            {uploadProgress < 100 ? (
              <span>Uploading... {uploadProgress}%</span>
            ) : (
              <span className="upload-progress-done">
                <CheckCircle2 size={14} />
                Processing complete
              </span>
            )}
          </div>
        </div>
      )}

      {/* Upload error (during upload) */}
      {uploadError && uploadedFile && (
        <div className="upload-error">
          <AlertCircle size={16} />
          <span>{uploadError}</span>
        </div>
      )}

      {/* Upload button */}
      {uploadedFile && !isUploading && (
        <button
          className="btn-primary upload-btn"
          onClick={handleUpload}
          disabled={isUploading}
        >
          <Upload size={16} />
          Upload &amp; Continue
        </button>
      )}
    </div>
  );
}
