"use client";

import { useEffect } from "react";
import { Cpu, AlertCircle, CheckCircle, Package } from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * GnnConfig — GNN/CHGNet transfer learning dependency status.
 * Shows installed/not-installed state with install instructions.
 */
export default function GnnConfig() {
  const { gnnStatus, gnnLoading, fetchGnnStatus } = useSettingsStore();

  useEffect(() => { fetchGnnStatus(); }, [fetchGnnStatus]);

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Cpu size={18} /></div>
        <div>
          <h3>GNN / CHGNet Transfer Learning</h3>
          <p className="settings-section-desc">
            Graph neural network support for crystal structure analysis (optional, heavy dependencies)
          </p>
        </div>
        {gnnStatus && (
          <span className={`settings-badge ${gnnStatus.installed ? "success" : "warning"}`}>
            {gnnStatus.installed ? "Installed" : "Not Installed"}
          </span>
        )}
      </div>

      {gnnLoading ? (
        <div className="settings-loading">Checking GNN dependencies...</div>
      ) : gnnStatus ? (
        <div className="gnn-status-panel">
          <div className="gnn-status-row">
            {gnnStatus.installed ? (
              <CheckCircle size={16} className="gnn-icon-ok" />
            ) : (
              <AlertCircle size={16} className="gnn-icon-warn" />
            )}
            <span>{gnnStatus.message}</span>
          </div>

          {gnnStatus.pytorch_version && (
            <div className="gnn-detail">
              <Package size={13} />
              <span>PyTorch: {gnnStatus.pytorch_version}</span>
            </div>
          )}

          {gnnStatus.chgnet_version && gnnStatus.chgnet_version !== "not installed" && (
            <div className="gnn-detail">
              <Package size={13} />
              <span>CHGNet: {gnnStatus.chgnet_version}</span>
            </div>
          )}

          {!gnnStatus.installed && gnnStatus.install_instructions && (
            <div className="gnn-install-section">
              <h4>Installation Instructions</h4>
              <p className="gnn-install-note">
                GNN dependencies are heavy (~2GB). Install only if you need crystal structure analysis.
                During <code>dev.sh setup</code>, select &quot;Accept&quot; for GNN dependencies.
              </p>
              <pre className="gnn-install-code">
                {gnnStatus.install_instructions}
              </pre>
            </div>
          )}

          {!gnnStatus.enabled && (
            <div className="gnn-disabled-note">
              <AlertCircle size={14} />
              GNN module is disabled. Set <code>ENABLE_GNN_MODULE=true</code> in .env to enable.
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
