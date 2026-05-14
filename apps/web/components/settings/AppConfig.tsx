"use client";

import { useEffect, useState, useRef } from "react";
import {
  Palette, Save, RotateCcw, Upload, FileText, AlertCircle,
  Image as ImageIcon, ChevronDown, ShieldCheck,
} from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";
import { useUIStore } from "@/lib/store/uiStore";
import InfoTooltip from "@/components/ui/InfoTooltip";

/**
 * AppConfig — Manage app branding and environment variables.
 * Merged App Branding + Frontend Public (synced automatically).
 * Includes .env import, logo upload, premium feature flag toggles.
 */

const FIELD_INFO: Record<string, string> = {
  APP_NAME: "Application display name. Auto-syncs to NEXT_PUBLIC_APP_NAME. Example: Piezo.AI",
  APP_VERSION: "Semantic version (major.minor.patch). Auto-syncs to NEXT_PUBLIC_APP_VERSION. Example: 2.1.0",
  APP_LOGO_TEXT: "Single character for the sidebar logo. Auto-syncs to NEXT_PUBLIC_APP_LOGO_TEXT. Example: P",
  APP_LOGO_PATH: "Path to logo image relative to /public. Auto-syncs to NEXT_PUBLIC_APP_LOGO_PATH. Example: /piezo-ai-logo.png",
  APP_TAGLINE: "Short description shown in the sidebar. Example: AI-Driven Material Discovery",
  NEXT_PUBLIC_DEV_NAME: "Developer name shown in footer. Example: John Doe",
  NEXT_PUBLIC_DEV_GITHUB: "GitHub profile URL. Example: https://github.com/username",
  NEXT_PUBLIC_DEV_LINKEDIN: "LinkedIn profile URL. Example: https://linkedin.com/in/username",
  ENABLE_COMPOSITE_MODULE: "Enable composite materials section. Values: true or false",
  ENABLE_HARDNESS_MODULE: "Enable hardness prediction target. Values: true or false",
  ENABLE_GNN_MODULE: "Enable GNN/CHGNet crystal analysis (requires heavy deps). Values: true or false",
  MODEL_ARTIFACTS_PATH: "Directory for trained model files. Example: ./resources/trained-models",
  TRAINING_ARTIFACTS_PATH: "Directory for training artifacts. Example: ./resources/training-artifacts",
};

const BOOL_FIELDS = new Set(["ENABLE_COMPOSITE_MODULE", "ENABLE_HARDNESS_MODULE", "ENABLE_GNN_MODULE"]);

// Synced pairs: changing one auto-updates the other
const SYNC_PAIRS: Record<string, string> = {
  APP_NAME: "NEXT_PUBLIC_APP_NAME",
  APP_VERSION: "NEXT_PUBLIC_APP_VERSION",
  APP_LOGO_TEXT: "NEXT_PUBLIC_APP_LOGO_TEXT",
  APP_LOGO_PATH: "NEXT_PUBLIC_APP_LOGO_PATH",
};

export default function AppConfig() {
  const { appConfig, configLoading, configSaving, fetchAppConfig, saveAppConfig, importEnvFile } = useSettingsStore();
  const { strictFormulaMode, setStrictFormulaMode } = useUIStore();
  const [localConfig, setLocalConfig] = useState<Record<string, string>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  // Logo preview
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  const [pendingLogoFile, setPendingLogoFile] = useState<File | null>(null);
  const logoInputRef = useRef<HTMLInputElement>(null);
  const [logoUploadResult, setLogoUploadResult] = useState<{ success: boolean; message: string } | null>(null);
  const [logoUploadError, setLogoUploadError] = useState<string | null>(null);

  // Import state
  const [importResult, setImportResult] = useState<{ success: boolean; message: string } | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchAppConfig(); }, [fetchAppConfig]);
  useEffect(() => {
    setLocalConfig({ ...appConfig });
    setHasChanges(false);
  }, [appConfig]);

  const handleChange = (key: string, value: string) => {
    const updates: Record<string, string> = { [key]: value };
    // Auto-sync paired fields
    if (SYNC_PAIRS[key]) updates[SYNC_PAIRS[key]] = value;
    setLocalConfig((prev) => ({ ...prev, ...updates }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    const changes: Record<string, string> = {};
    for (const [key, value] of Object.entries(localConfig)) {
      if (value !== appConfig[key]) {
        changes[key] = value;
      }
    }
    if (Object.keys(changes).length > 0) {
      await saveAppConfig(changes);
    }
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalConfig({ ...appConfig });
    setHasChanges(false);
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const isValid = file.name.startsWith(".env");
    if (!isValid) {
      setImportError(`Invalid file: "${file.name}". Only .env and .env.* files are accepted.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    if (file.size > 100_000) {
      setImportError("File too large. .env files should be under 100KB.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }
    setImporting(true);
    setImportError(null);
    setImportResult(null);
    try {
      const result = await importEnvFile(file);
      setImportResult(result);
    } catch (err: any) {
      setImportError(err.message || "Import failed");
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleLogoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setLogoUploadError("Only image files are allowed for logo upload.");
      return;
    }
    if (file.size > 2_000_000) {
      setLogoUploadError("Logo file too large. Maximum 2MB.");
      return;
    }
    
    setPendingLogoFile(file);
    const url = URL.createObjectURL(file);
    setLogoPreview(url);
    setLogoUploadError(null);
    setLogoUploadResult(null);
  };

  const confirmLogoUpload = async () => {
    if (!pendingLogoFile) return;
    try {
      setLogoUploadError(null);
      const { uploadLogo } = await import("@/lib/api/settings");
      const result = await uploadLogo(pendingLogoFile);
      await fetchAppConfig(); // Fetch updated config from backend to update all UI
      handleChange("APP_LOGO_PATH", result.path);
      setLogoUploadResult({ success: true, message: result.message });
      setPendingLogoFile(null);
      setLogoPreview(null);
    } catch (err: any) {
      setLogoUploadError(err.message || "Logo upload failed");
    }
  };

  const cancelLogoUpload = () => {
    setPendingLogoFile(null);
    setLogoPreview(null);
    setLogoUploadError(null);
    setLogoUploadResult(null);
    if (logoInputRef.current) logoInputRef.current.value = "";
  };

  const groups = [
    {
      title: "App Branding",
      desc: "Name, version, and logo are auto-synced to NEXT_PUBLIC_* frontend variables.",
      fields: [
        { key: "APP_NAME", label: "App Name", placeholder: "Piezo.AI" },
        { key: "APP_VERSION", label: "Version", placeholder: "2.1.0" },
        { key: "APP_LOGO_TEXT", label: "Logo Text", placeholder: "P" },
        { key: "APP_LOGO_PATH", label: "Logo Path", placeholder: "/piezo-ai-logo.png", hasLogoUpload: true },
        { key: "APP_TAGLINE", label: "Tagline", placeholder: "AI-Driven Piezoelectric Material Discovery" },
      ],
    },
    {
      title: "Developer Info",
      fields: [
        { key: "NEXT_PUBLIC_DEV_NAME", label: "Developer Name", placeholder: "Your Name" },
        { key: "NEXT_PUBLIC_DEV_GITHUB", label: "GitHub URL", placeholder: "https://github.com/username" },
        { key: "NEXT_PUBLIC_DEV_LINKEDIN", label: "LinkedIn URL", placeholder: "https://linkedin.com/in/username" },
      ],
    },
    {
      title: "Feature Flags",
      fields: [
        { key: "ENABLE_COMPOSITE_MODULE", label: "Composite Module", placeholder: "true" },
        { key: "ENABLE_HARDNESS_MODULE", label: "Hardness Module", placeholder: "true" },
        { key: "ENABLE_GNN_MODULE", label: "GNN Module", placeholder: "false" },
      ],
    },
    {
      title: "Paths",
      fields: [
        { key: "MODEL_ARTIFACTS_PATH", label: "Model Artifacts", placeholder: "./resources/trained-models" },
        { key: "TRAINING_ARTIFACTS_PATH", label: "Training Artifacts", placeholder: "./resources/training-artifacts" },
      ],
    },
  ];

  const currentLogoPath = localConfig["APP_LOGO_PATH"] || "/piezo-ai-logo.png";

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Palette size={18} /></div>
        <div>
          <h3>App Environment Configuration</h3>
          <p className="settings-section-desc">
            Manage branding, version, developer info, and feature flags. Changes are written to .env.
          </p>
        </div>
      </div>

      <div className="config-priority-note">
        <strong>Priority order:</strong> Terminal env vars → .env file → Application defaults
      </div>

      {/* Import .env file */}
      <div className="config-import-section">
        <div className="config-import-header">
          <Upload size={14} />
          <strong>Import .env File</strong>
          <span className="config-import-info">
            Upload a .env file to bulk-update configuration. Sensitive keys are skipped.
          </span>
        </div>
        <div className="config-import-row">
          <label className="config-import-btn" htmlFor="env-import-input">
            <FileText size={14} />
            {importing ? "Importing..." : "Choose .env File"}
          </label>
          <input id="env-import-input" ref={fileInputRef} type="file" accept=".env,.env.*"
            onChange={handleImport} style={{ display: "none" }} disabled={importing} />
          <span className="config-import-hint">Accepts: .env, .env.local, .env.example, etc.</span>
        </div>
        {importError && (
          <div className="config-import-error">
            <AlertCircle size={13} /> {importError}
          </div>
        )}
        {importResult && (
          <div className={`config-import-result ${importResult.success ? "success" : "error"}`}>
            {importResult.message}
          </div>
        )}
      </div>

      {/* Strict Formula Validation Toggle */}
      <div className="strict-mode-card">
        <div className="strict-mode-info">
          <ShieldCheck size={16} className="strict-mode-icon" />
          <div>
            <div className="strict-mode-label">Strict Formula Validation</div>
            <div className="strict-mode-desc">
              {strictFormulaMode
                ? "Enforces element casing, bracket rules, and charset restrictions"
                : "Legacy mode — accepts most formula formats"}
            </div>
          </div>
        </div>
        <button
          className={`toggle-switch ${strictFormulaMode ? "active" : ""}`}
          onClick={() => setStrictFormulaMode(!strictFormulaMode)}
          role="switch"
          aria-checked={strictFormulaMode}
          aria-label="Toggle strict formula validation"
        >
          <span className="toggle-knob" />
        </button>
      </div>

      {configLoading ? (
        <div className="settings-loading">Loading configuration...</div>
      ) : (
        <div className="config-groups">
          {groups.map((group) => (
            <div key={group.title} className="config-group">
              <h4 className="config-group-title">{group.title}</h4>
              {"desc" in group && group.desc && (
                <p className="config-group-desc">{group.desc}</p>
              )}
              <div className="config-fields">
                {group.fields.map((field) => (
                  <div key={field.key} className="config-field">
                    <div className="config-field-label-row">
                      <label htmlFor={field.key}>{field.label}</label>
                      {FIELD_INFO[field.key] && (
                        <InfoTooltip text={FIELD_INFO[field.key]} />
                      )}
                    </div>

                    {/* Logo path field with upload + preview */}
                    {"hasLogoUpload" in field && field.hasLogoUpload ? (
                      <div className="config-logo-field">
                        <div className="config-logo-preview">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={logoPreview || currentLogoPath}
                            alt="Logo preview"
                            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                          />
                        </div>
                        <div className="config-logo-inputs">
                          <input
                            id={field.key}
                            value={localConfig[field.key] || ""}
                            onChange={(e) => handleChange(field.key, e.target.value)}
                            placeholder={field.placeholder}
                            className="config-input"
                          />
                          <button className="config-logo-upload-btn"
                            onClick={() => logoInputRef.current?.click()}>
                            <ImageIcon size={12} /> {pendingLogoFile ? "Change File" : "Upload Logo"}
                          </button>
                          <input id="app-logo-upload-input" name="app_logo_upload" ref={logoInputRef} type="file" accept="image/*"
                            onChange={handleLogoSelect} style={{ display: "none" }} />
                        </div>
                        {pendingLogoFile && (
                          <div style={{ display: "flex", gap: "8px", marginTop: "12px", width: "100%", paddingLeft: "52px" }}>
                            <button 
                              onClick={confirmLogoUpload}
                              style={{ flex: 1, padding: "6px", background: "var(--primary)", color: "#fff", borderRadius: "6px", fontSize: "12px", border: "none", cursor: "pointer" }}
                            >
                              Save Logo
                            </button>
                            <button 
                              onClick={cancelLogoUpload}
                              style={{ flex: 1, padding: "6px", background: "transparent", color: "var(--text)", borderRadius: "6px", fontSize: "12px", border: "1px solid var(--border)", cursor: "pointer" }}
                            >
                              Cancel
                            </button>
                          </div>
                        )}
                        {logoUploadError && (
                          <div className="config-import-error" style={{ marginTop: "12px", width: "100%", marginLeft: "52px" }}>
                            <AlertCircle size={13} /> {logoUploadError}
                          </div>
                        )}
                        {logoUploadResult && (
                          <div className={`config-import-result ${logoUploadResult.success ? "success" : "error"}`} style={{ marginTop: "12px", width: "100%", marginLeft: "52px" }}>
                            {logoUploadResult.message}
                          </div>
                        )}
                      </div>
                    ) : BOOL_FIELDS.has(field.key) ? (
                      /* Premium toggle dropdown for boolean fields */
                      <div className="config-toggle-wrap">
                        <button
                          id={field.key}
                          className={`config-toggle-btn ${localConfig[field.key] === "true" ? "active" : ""}`}
                          onClick={() => setOpenDropdown(openDropdown === field.key ? null : field.key)}
                        >
                          <span className={`config-toggle-dot ${localConfig[field.key] === "true" ? "on" : "off"}`} />
                          <span className="config-toggle-label">
                            {localConfig[field.key] === "true" ? "Enabled" : "Disabled"}
                          </span>
                          <ChevronDown size={12} className={`config-toggle-chevron ${openDropdown === field.key ? "open" : ""}`} />
                        </button>
                        {openDropdown === field.key && (
                          <div className="config-toggle-dropdown">
                            <button
                              className={`config-toggle-option ${localConfig[field.key] === "true" ? "selected" : ""}`}
                              onClick={() => { handleChange(field.key, "true"); setOpenDropdown(null); }}
                            >
                              <span className="config-toggle-dot on" />
                              Enabled
                              {localConfig[field.key] === "true" && <span className="config-toggle-check">✓</span>}
                            </button>
                            <button
                              className={`config-toggle-option ${localConfig[field.key] !== "true" ? "selected" : ""}`}
                              onClick={() => { handleChange(field.key, "false"); setOpenDropdown(null); }}
                            >
                              <span className="config-toggle-dot off" />
                              Disabled
                              {localConfig[field.key] !== "true" && <span className="config-toggle-check">✓</span>}
                            </button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <input
                        id={field.key}
                        value={localConfig[field.key] || ""}
                        onChange={(e) => handleChange(field.key, e.target.value)}
                        placeholder={field.placeholder}
                        className="config-input"
                      />
                    )}
                    <span className="config-key">{field.key}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {hasChanges && (
        <div className="config-actions">
          <button className="config-save-btn" onClick={handleSave} disabled={configSaving}>
            <Save size={14} /> {configSaving ? "Saving..." : "Save Changes"}
          </button>
          <button className="config-reset-btn" onClick={handleReset}>
            <RotateCcw size={14} /> Reset
          </button>
        </div>
      )}
    </div>
  );
}
