"use client";

import { useState } from "react";
import { AlertTriangle, Trash2, Database, RotateCcw, ShieldAlert } from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * DangerZone — Destructive actions (purge models, clear cache, factory reset).
 * Each action requires explicit confirmation.
 */
export default function DangerZone() {
  const { purgeAllModels, clearCache, resetAllSettings } = useSettingsStore();
  const [purgeConfirm, setPurgeConfirm] = useState(false);
  const [cacheConfirm, setCacheConfirm] = useState(false);
  const [resetConfirm, setResetConfirm] = useState(false);
  const [resetDoubleConfirm, setResetDoubleConfirm] = useState(false);
  const [purging, setPurging] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handlePurge = async () => {
    setPurging(true);
    try {
      const res = await purgeAllModels();
      setResult(res.message);
    } catch (e: any) {
      setResult(`Error: ${e.message}`);
    }
    setPurging(false);
    setPurgeConfirm(false);
  };

  const handleClear = async () => {
    setClearing(true);
    try {
      const res = await clearCache();
      setResult(res.message);
    } catch (e: any) {
      setResult(`Error: ${e.message}`);
    }
    setClearing(false);
    setCacheConfirm(false);
  };

  const handleReset = async () => {
    setResetting(true);
    try {
      const res = await resetAllSettings();
      setResult(res.message + " Actions: " + res.actions_taken.join("; "));
    } catch (e: any) {
      setResult(`Error: ${e.message}`);
    }
    setResetting(false);
    setResetConfirm(false);
    setResetDoubleConfirm(false);
  };

  return (
    <div className="settings-section danger-zone">
      <div className="settings-section-header">
        <div className="settings-section-icon danger">
          <AlertTriangle size={18} />
        </div>
        <div>
          <h3>Danger Zone</h3>
          <p className="settings-section-desc">
            Destructive actions — cannot be undone. Proceed with caution.
          </p>
        </div>
      </div>

      {result && (
        <div className="danger-result">
          <RotateCcw size={14} />
          {result}
          <button onClick={() => setResult(null)} className="danger-result-close">×</button>
        </div>
      )}

      <div className="danger-actions">
        {/* Purge All Models */}
        <div className="danger-action-card">
          <div className="danger-action-info">
            <Trash2 size={16} />
            <div>
              <strong>Purge All Models</strong>
              <p>Delete all trained models from database and filesystem. Training artifacts and predictions will also be removed.</p>
            </div>
          </div>
          {purgeConfirm ? (
            <div className="danger-confirm-row">
              <span className="danger-confirm-text">Are you sure? This cannot be undone.</span>
              <button className="danger-confirm-btn yes" onClick={handlePurge} disabled={purging}>
                {purging ? "Purging..." : "Yes, Purge All"}
              </button>
              <button className="danger-confirm-btn no" onClick={() => setPurgeConfirm(false)}>Cancel</button>
            </div>
          ) : (
            <button className="danger-action-btn" onClick={() => setPurgeConfirm(true)}>
              <Trash2 size={14} /> Purge Models
            </button>
          )}
        </div>

        {/* Clear Prediction Cache */}
        <div className="danger-action-card">
          <div className="danger-action-info">
            <Database size={16} />
            <div>
              <strong>Clear Prediction Cache</strong>
              <p>Remove all cached prediction results and generated reports from the database.</p>
            </div>
          </div>
          {cacheConfirm ? (
            <div className="danger-confirm-row">
              <span className="danger-confirm-text">Are you sure? All prediction history will be lost.</span>
              <button className="danger-confirm-btn yes" onClick={handleClear} disabled={clearing}>
                {clearing ? "Clearing..." : "Yes, Clear All"}
              </button>
              <button className="danger-confirm-btn no" onClick={() => setCacheConfirm(false)}>Cancel</button>
            </div>
          ) : (
            <button className="danger-action-btn" onClick={() => setCacheConfirm(true)}>
              <Database size={14} /> Clear Cache
            </button>
          )}
        </div>

        {/* Factory Reset — All Settings */}
        <div className="danger-action-card factory-reset-card">
          <div className="danger-action-info">
            <ShieldAlert size={16} />
            <div>
              <strong>Factory Reset — All Settings</strong>
              <p>
                Restore <strong>everything</strong> to original defaults: .env file, element registry,
                custom properties, feature flags, LLM config. Does NOT delete trained models or prediction data.
                The server will need a restart after this action for full effect.
              </p>
            </div>
          </div>
          {resetConfirm ? (
            resetDoubleConfirm ? (
              <div className="danger-confirm-row">
                <span className="danger-confirm-text danger-confirm-critical">
                  ⚠️ FINAL CONFIRMATION — This resets .env, elements, properties, and all customizations.
                  You will need to reconfigure LLM keys and restart the server.
                </span>
                <button className="danger-confirm-btn yes" onClick={handleReset} disabled={resetting}>
                  {resetting ? "Resetting..." : "Yes, Factory Reset"}
                </button>
                <button className="danger-confirm-btn no"
                  onClick={() => { setResetDoubleConfirm(false); setResetConfirm(false); }}>
                  Cancel
                </button>
              </div>
            ) : (
              <div className="danger-confirm-row">
                <span className="danger-confirm-text">This will reset ALL settings. Are you sure?</span>
                <button className="danger-confirm-btn yes" onClick={() => setResetDoubleConfirm(true)}>
                  Continue
                </button>
                <button className="danger-confirm-btn no" onClick={() => setResetConfirm(false)}>Cancel</button>
              </div>
            )
          ) : (
            <button className="danger-action-btn factory-reset-btn" onClick={() => setResetConfirm(true)}>
              <ShieldAlert size={14} /> Factory Reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
