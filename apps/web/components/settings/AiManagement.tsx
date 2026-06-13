"use client";

import { useEffect, useState } from "react";
import {
  Sparkles, Bot, Brain, Search, Server as ServerIcon,
  Settings as SettingsIcon, Save, ChevronDown, ChevronUp,
  Check, Key, Globe, Thermometer, Hash, Eye, EyeOff,
} from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";
import InfoTooltip from "@/components/ui/InfoTooltip";

const PROVIDER_ICONS: Record<string, React.ElementType> = {
  google: Sparkles, openai: Bot, anthropic: Brain,
  deepseek: Search, ollama: ServerIcon, custom: SettingsIcon,
};

const FIELD_TIPS: Record<string, string> = {
  model: "The specific model variant to use. Select from presets or enter a custom model name.",
  api_key: "Your provider API key. This is stored in .env and never exposed in the UI. Enter a new value to update.",
  base_url: "API endpoint URL. Required for DeepSeek, Ollama, and custom providers. Example: http://localhost:11434",
  temperature: "Controls randomness (0 = deterministic, 2 = very creative). Recommended: 0.1–0.3 for scientific analysis.",
  max_tokens: "Maximum output length in tokens. Range: 256–128000. Recommended: 4096 for reports.",
};

/**
 * AiManagement — Full LLM provider management UI.
 * Reads initial state from .env to show correct configured status.
 */
export default function AiManagement() {
  const {
    llmConfig, llmProviders, llmLoading, llmSaving,
    fetchLlmConfig, fetchLlmProviders, saveLlmConfig,
  } = useSettingsStore();

  const [selectedProvider, setSelectedProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [temperature, setTemperature] = useState(0.1);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [customModel, setCustomModel] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    fetchLlmConfig();
    fetchLlmProviders();
  }, [fetchLlmConfig, fetchLlmProviders]);

  useEffect(() => {
    if (llmConfig) {
      setSelectedProvider(llmConfig.provider);
      setModel(llmConfig.model);
      setBaseUrl(llmConfig.base_url);
      setTemperature(llmConfig.temperature);
      setMaxTokens(llmConfig.max_tokens);
    }
  }, [llmConfig]);

  const currentProvider = llmProviders.find((p) => p.id === selectedProvider);

  const handleSave = async () => {
    const finalModel = customModel || model;
    await saveLlmConfig({
      provider: selectedProvider,
      model: finalModel,
      ...(apiKey ? { api_key: apiKey } : {}),
      base_url: baseUrl,
      temperature,
      max_tokens: maxTokens,
    });
    setApiKey("");
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const statusColor = llmConfig?.status === "ready" ? "var(--success)"
    : llmConfig?.status === "error" ? "var(--error)" : "var(--text-muted)";

  const InfoTip = ({ field }: { field: string }) => (
    <InfoTooltip text={FIELD_TIPS[field]} />
  );

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Sparkles size={18} /></div>
        <div>
          <h3>AI / LLM Management</h3>
          <p className="settings-section-desc">
            Configure AI providers for report insights and analysis
          </p>
        </div>
        <div className="llm-status-badge" style={{ color: statusColor, borderColor: statusColor }}>
          <span className="sys-flag-dot" style={{ background: statusColor }} />
          {llmConfig?.status_message || "Not configured"}
        </div>
      </div>

      {llmLoading ? (
        <div className="settings-loading">Loading AI config...</div>
      ) : (
        <>
          {/* Provider Selection */}
          <div className="llm-providers-grid">
            {llmProviders.map((p) => {
              const Icon = PROVIDER_ICONS[p.id] || SettingsIcon;
              return (
                <button
                  key={p.id}
                  className={`llm-provider-card ${selectedProvider === p.id ? "selected" : ""}`}
                  onClick={() => {
                    setSelectedProvider(p.id);
                    // If switching to the currently configured provider, use its saved model
                    if (p.id === llmConfig?.provider) {
                      setModel(llmConfig.model || p.default_models[0] || "");
                    } else {
                      setModel(p.default_models[0] || "");
                    }
                    setCustomModel("");
                  }}
                >
                  <Icon size={20} />
                  <span className="llm-provider-name">{p.name}</span>
                  <span className="llm-provider-desc">{p.description}</span>
                  {selectedProvider === p.id && (
                    <span className="llm-provider-check"><Check size={12} /></span>
                  )}
                </button>
              );
            })}
          </div>

          {selectedProvider && currentProvider && (
            <div className="llm-config-panel">
              {/* Model Selection */}
              <div className="llm-field">
                <label><Hash size={13} /> Model <InfoTip field="model" /></label>
                {currentProvider.default_models.length > 0 ? (
                  <div className="llm-model-selector">
                    <select
                      value={model}
                      onChange={(e) => { setModel(e.target.value); setCustomModel(""); }}
                      className="llm-select"
                    >
                      {currentProvider.default_models.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                      <option value="__custom__">Custom model...</option>
                    </select>
                    {model === "__custom__" && (
                      <input
                        className="llm-input"
                        value={customModel}
                        onChange={(e) => setCustomModel(e.target.value)}
                        placeholder="Enter model name (e.g., my-model-v2)"
                      />
                    )}
                  </div>
                ) : (
                  <input
                    className="llm-input"
                    value={customModel || model}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="Enter model name"
                  />
                )}
              </div>

              {/* API Key */}
              {currentProvider.requires_api_key && (
                <div className="llm-field">
                  <label><Key size={13} /> API Key <InfoTip field="api_key" /></label>
                  <div className="llm-key-row">
                    <input
                      className="llm-input"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={llmConfig?.has_api_key ? "••••••••  (key set — enter new to update)" : "Enter API key"}
                    />
                    <button
                      className="llm-eye-btn"
                      onClick={() => setShowApiKey(!showApiKey)}
                      title={showApiKey ? "Hide API key" : "Show API key"}
                      type="button"
                    >
                      {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                </div>
              )}

              {/* Base URL */}
              {currentProvider.requires_base_url && (
                <div className="llm-field">
                  <label><Globe size={13} /> Base URL / Endpoint <InfoTip field="base_url" /></label>
                  <input
                    className="llm-input"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder={selectedProvider === "ollama" ? "http://localhost:11434" : "https://api.example.com/v1"}
                  />
                </div>
              )}

              {/* Advanced Settings */}
              <button
                className="llm-advanced-toggle"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                Advanced Settings
              </button>

              {showAdvanced && (
                <div className="llm-advanced-panel">
                  <div className="llm-field">
                    <label><Thermometer size={13} /> Temperature <InfoTip field="temperature" /></label>
                    <div className="llm-slider-row">
                      <input
                        type="range"
                        min="0" max="2" step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="llm-slider"
                      />
                      <span className="llm-slider-value">{temperature}</span>
                    </div>
                  </div>
                  <div className="llm-field">
                    <label><Hash size={13} /> Max Tokens <InfoTip field="max_tokens" /></label>
                    <input
                      type="number"
                      className="llm-input"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 4096)}
                      min={256} max={128000}
                    />
                  </div>
                </div>
              )}

              {/* Save */}
              <button
                className="llm-save-btn"
                onClick={handleSave}
                disabled={llmSaving}
              >
                {saved ? <><Check size={14} /> Saved!</> : (
                  <><Save size={14} /> {llmSaving ? "Saving..." : "Save AI Configuration"}</>
                )}
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
