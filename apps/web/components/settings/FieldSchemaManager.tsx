"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Database, Plus, Trash2, RefreshCw, RotateCcw, X, ChevronDown,
  ChevronRight, Download, Upload, Tag, Hash, Type, Layers, Info,
  AlertCircle, Search, UserPlus,
} from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";
import type { FieldDefinition } from "@/lib/api/settings";

/* ── Helpers ─────────────────────── */

const TYPE_COLORS: Record<string, string> = {
  float: "#10B981",
  int: "#6366F1",
  string: "#F59E0B",
  category: "#8B5CF6",
};

const TYPE_ICONS: Record<string, typeof Hash> = {
  float: Hash,
  int: Hash,
  string: Type,
  category: Tag,
};

/* ── Component ─────────────────────── */

export default function FieldSchemaManager() {
  const {
    fieldSchema, fieldSchemaLoading, fetchFieldSchema,
    addField, removeField, addCategoryValue, removeCategoryValue,
    exportSchema, importSchema,
  } = useSettingsStore();

  const [statusMsg, setStatusMsg] = useState<{ text: string; type: "success" | "error" } | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedField, setExpandedField] = useState<string | null>(null);
  const [showAddField, setShowAddField] = useState(false);
  const [showAddCategory, setShowAddCategory] = useState<string | null>(null);
  const [newCatValue, setNewCatValue] = useState("");
  const [addingCat, setAddingCat] = useState(false);

  // Add field form state
  const [newFieldName, setNewFieldName] = useState("");
  const [newFieldType, setNewFieldType] = useState("float");
  const [newFieldDesc, setNewFieldDesc] = useState("");
  const [newFieldIsTarget, setNewFieldIsTarget] = useState(false);
  const [newFieldIsComposite, setNewFieldIsComposite] = useState(false);
  const [newFieldCatValues, setNewFieldCatValues] = useState("");
  const [newFieldRangeMin, setNewFieldRangeMin] = useState("");
  const [newFieldRangeMax, setNewFieldRangeMax] = useState("");
  const [addingField, setAddingField] = useState(false);
  const [fieldNameError, setFieldNameError] = useState("");

  // Confirm states
  const [confirmRemoveField, setConfirmRemoveField] = useState<string | null>(null);
  const [confirmRemoveCat, setConfirmRemoveCat] = useState<{ field: string; value: string } | null>(null);

  useEffect(() => { fetchFieldSchema(); }, [fetchFieldSchema]);

  /* ── Filtering ─────────────────────── */

  const filteredFields = (fieldSchema || []).filter((f) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return f.name.includes(q) || f.description.toLowerCase().includes(q) || f.data_type.includes(q);
  });

  const categoricalFields = filteredFields.filter((f) => f.data_type === "category");
  const numericFields = filteredFields.filter((f) => f.data_type === "float" || f.data_type === "int");
  const otherFields = filteredFields.filter((f) => f.data_type === "string");

  /* ── Handlers ─────────────────────── */

  const validateFieldName = (name: string) => {
    if (!name) { setFieldNameError(""); return; }
    if (!/^[a-z][a-z0-9_]*$/.test(name)) {
      setFieldNameError("Must be snake_case (lowercase letters, numbers, underscores)");
    } else {
      setFieldNameError("");
    }
  };

  const handleAddField = async () => {
    if (!newFieldName.trim() || fieldNameError) return;
    setAddingField(true);
    try {
      const catValues = newFieldType === "category" && newFieldCatValues.trim()
        ? newFieldCatValues.split(",").map((v) => v.trim().toLowerCase().replace(/\s+/g, "_")).filter(Boolean)
        : [];
      await addField({
        name: newFieldName.trim(),
        data_type: newFieldType,
        description: newFieldDesc.trim(),
        is_target: newFieldIsTarget,
        is_composite_field: newFieldIsComposite,
        category_values: catValues,
        range_min: newFieldRangeMin ? parseFloat(newFieldRangeMin) : null,
        range_max: newFieldRangeMax ? parseFloat(newFieldRangeMax) : null,
      });
      setStatusMsg({ text: `Field "${newFieldName}" added successfully`, type: "success" });
      setNewFieldName(""); setNewFieldDesc(""); setNewFieldCatValues("");
      setNewFieldRangeMin(""); setNewFieldRangeMax("");
      setNewFieldIsTarget(false); setNewFieldIsComposite(false);
      setShowAddField(false);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Failed to add field", type: "error" });
    }
    setAddingField(false);
  };

  const handleRemoveField = async (name: string) => {
    try {
      await removeField(name);
      setStatusMsg({ text: `Field "${name}" removed`, type: "success" });
      setConfirmRemoveField(null);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Cannot remove field", type: "error" });
      setConfirmRemoveField(null);
    }
  };

  const handleAddCategory = async (fieldName: string) => {
    if (!newCatValue.trim()) return;
    setAddingCat(true);
    try {
      await addCategoryValue(fieldName, newCatValue.trim());
      setStatusMsg({ text: `"${newCatValue}" added to ${fieldName}`, type: "success" });
      setNewCatValue("");
      setShowAddCategory(null);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Failed to add category", type: "error" });
    }
    setAddingCat(false);
  };

  const handleRemoveCategory = async (fieldName: string, value: string) => {
    try {
      await removeCategoryValue(fieldName, value);
      setStatusMsg({ text: `"${value}" removed from ${fieldName}`, type: "success" });
      setConfirmRemoveCat(null);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Cannot remove built-in category", type: "error" });
      setConfirmRemoveCat(null);
    }
  };

  const handleExport = async () => {
    try {
      const data = await exportSchema();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `piezo-field-schema-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setStatusMsg({ text: "Field schema exported", type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Export failed", type: "error" });
    }
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      await importSchema(data);
      setStatusMsg({ text: "Field schema imported successfully", type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Import failed", type: "error" });
    }
    e.target.value = "";
  };

  /* ── Render helpers ─────────────────────── */

  const renderFieldRow = (field: FieldDefinition) => {
    const isExpanded = expandedField === field.name;
    const TypeIcon = TYPE_ICONS[field.data_type] || Hash;
    const typeColor = TYPE_COLORS[field.data_type] || "#6B7280";

    return (
      <div key={field.name} className="field-schema-row">
        <div
          className="field-schema-row-header"
          onClick={() => setExpandedField(isExpanded ? null : field.name)}
        >
          <div className="field-schema-row-left">
            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="field-schema-name">{field.name}</span>
            <span
              className="field-schema-type-badge"
              style={{ background: `${typeColor}18`, color: typeColor, borderColor: `${typeColor}40` }}
            >
              <TypeIcon size={10} />
              {field.data_type}
            </span>
            {field.is_target && (
              <span className="field-schema-target-badge">target</span>
            )}
            {field.is_composite_field && (
              <span className="field-schema-composite-badge">composite</span>
            )}
            {field.is_user_added && (
              <span className="field-schema-user-badge" title="User-added field">
                <UserPlus size={9} />
              </span>
            )}
          </div>
          <div className="field-schema-row-right">
            {field.data_type === "category" && (
              <span className="field-schema-cat-count">{field.category_values.length} values</span>
            )}
            {field.is_user_added && (
              confirmRemoveField === field.name ? (
                <span className="field-schema-confirm-inline">
                  <button className="field-schema-confirm-yes" onClick={(e) => { e.stopPropagation(); handleRemoveField(field.name); }}>✓</button>
                  <button className="field-schema-confirm-no" onClick={(e) => { e.stopPropagation(); setConfirmRemoveField(null); }}>✗</button>
                </span>
              ) : (
                <button
                  className="field-schema-remove-btn"
                  onClick={(e) => { e.stopPropagation(); setConfirmRemoveField(field.name); }}
                  title="Remove user-added field"
                >
                  <X size={12} />
                </button>
              )
            )}
          </div>
        </div>

        {isExpanded && (
          <div className="field-schema-details">
            {field.description && (
              <p className="field-schema-desc">{field.description}</p>
            )}

            {(field.range_min !== null || field.range_max !== null) && (
              <div className="field-schema-range">
                <span className="field-schema-detail-label">Range:</span>
                {field.range_min !== null ? field.range_min : "−∞"} — {field.range_max !== null ? field.range_max : "∞"}
              </div>
            )}

            {field.data_type === "category" && (
              <div className="field-schema-categories">
                <div className="field-schema-cat-header">
                  <span className="field-schema-detail-label">
                    Allowed Values ({field.category_values.length})
                  </span>
                  <button
                    className="field-schema-add-cat-btn"
                    onClick={() => setShowAddCategory(showAddCategory === field.name ? null : field.name)}
                  >
                    <Plus size={11} /> Add
                  </button>
                </div>

                {showAddCategory === field.name && (
                  <div className="field-schema-add-cat-row">
                    <input
                      className="field-schema-add-cat-input"
                      value={newCatValue}
                      onChange={(e) => setNewCatValue(e.target.value)}
                      placeholder="new_value"
                      onKeyDown={(e) => e.key === "Enter" && handleAddCategory(field.name)}
                    />
                    <button
                      className="field-schema-add-cat-confirm"
                      onClick={() => handleAddCategory(field.name)}
                      disabled={!newCatValue.trim() || addingCat}
                    >
                      {addingCat ? "..." : <Plus size={12} />}
                    </button>
                  </div>
                )}

                <div className="field-schema-cat-pills">
                  {field.category_values.map((val) => (
                    <span key={val} className="field-schema-cat-pill">
                      {val}
                      {confirmRemoveCat?.field === field.name && confirmRemoveCat?.value === val ? (
                        <span className="field-schema-cat-confirm">
                          <button onClick={() => handleRemoveCategory(field.name, val)} title="Confirm">✓</button>
                          <button onClick={() => setConfirmRemoveCat(null)} title="Cancel">✗</button>
                        </span>
                      ) : (
                        <button
                          className="field-schema-cat-remove"
                          onClick={() => setConfirmRemoveCat({ field: field.name, value: val })}
                          title="Remove category (user-added only)"
                        >
                          <X size={8} />
                        </button>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {Object.keys(field.aliases).length > 0 && (
              <div className="field-schema-aliases">
                <span className="field-schema-detail-label">Aliases:</span>
                <div className="field-schema-alias-list">
                  {Object.entries(field.aliases).map(([alias, canonical]) => (
                    <span key={alias} className="field-schema-alias-tag">
                      {alias} → {canonical}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {field.default_value && (
              <div className="field-schema-default">
                <span className="field-schema-detail-label">Default:</span> {field.default_value}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  /* ── Main render ─────────────────────── */

  const totalFields = fieldSchema?.length || 0;
  const catCount = fieldSchema?.filter((f) => f.data_type === "category").length || 0;
  const targetCount = fieldSchema?.filter((f) => f.is_target).length || 0;

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Database size={18} /></div>
        <div>
          <h3>Field Schema Manager</h3>
          <p className="settings-section-desc">
            {totalFields} fields · {catCount} categorical · {targetCount} targets
          </p>
        </div>
        <div className="settings-section-header-actions">
          <button
            className="settings-reset-btn-text"
            onClick={handleExport}
            title="Export field schema"
          >
            <Download size={13} /> Export
          </button>
          <label className="settings-reset-btn-text" title="Import field schema">
            <Upload size={13} /> Import
            <input
              type="file"
              accept=".json"
              onChange={handleImport}
              style={{ display: "none" }}
            />
          </label>
          <button className="settings-refresh-btn" onClick={fetchFieldSchema} title="Refresh">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Status message */}
      {statusMsg && (
        <div className={`elem-status-msg ${statusMsg.type}`}>
          {statusMsg.type === "error" && <AlertCircle size={13} />}
          {statusMsg.text}
          <button onClick={() => setStatusMsg(null)} className="danger-result-close">×</button>
        </div>
      )}

      {/* Search */}
      <div className="field-schema-search">
        <Search size={14} />
        <input
          className="field-schema-search-input"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search fields..."
        />
        {searchQuery && (
          <button className="field-schema-search-clear" onClick={() => setSearchQuery("")}>
            <X size={12} />
          </button>
        )}
      </div>

      {/* Add New Field */}
      <div className="field-schema-add-section">
        <button
          className={`field-schema-add-toggle ${showAddField ? "active" : ""}`}
          onClick={() => setShowAddField(!showAddField)}
        >
          <Plus size={14} /> Add New Field
        </button>

        {showAddField && (
          <div className="field-schema-add-form">
            <div className="field-schema-add-row">
              <div className="field-schema-add-group">
                <label>Field Name</label>
                <input
                  className="field-schema-add-input"
                  value={newFieldName}
                  onChange={(e) => { setNewFieldName(e.target.value); validateFieldName(e.target.value); }}
                  placeholder="e.g., dielectric_constant"
                />
                {fieldNameError && (
                  <span className="field-schema-field-error">{fieldNameError}</span>
                )}
              </div>
              <div className="field-schema-add-group">
                <label>Data Type</label>
                <select
                  className="field-schema-add-select"
                  value={newFieldType}
                  onChange={(e) => setNewFieldType(e.target.value)}
                >
                  <option value="float">Float (numeric)</option>
                  <option value="int">Integer</option>
                  <option value="string">Text</option>
                  <option value="category">Category</option>
                </select>
              </div>
            </div>

            <div className="field-schema-add-row">
              <div className="field-schema-add-group full">
                <label>Description</label>
                <input
                  className="field-schema-add-input"
                  value={newFieldDesc}
                  onChange={(e) => setNewFieldDesc(e.target.value)}
                  placeholder="What this field measures..."
                />
              </div>
            </div>

            {newFieldType === "category" && (
              <div className="field-schema-add-row">
                <div className="field-schema-add-group full">
                  <label>Category Values (comma-separated)</label>
                  <input
                    className="field-schema-add-input"
                    value={newFieldCatValues}
                    onChange={(e) => setNewFieldCatValues(e.target.value)}
                    placeholder="value1, value2, value3"
                  />
                </div>
              </div>
            )}

            {(newFieldType === "float" || newFieldType === "int") && (
              <div className="field-schema-add-row">
                <div className="field-schema-add-group">
                  <label>Min Range</label>
                  <input
                    type="number"
                    className="field-schema-add-input"
                    value={newFieldRangeMin}
                    onChange={(e) => setNewFieldRangeMin(e.target.value)}
                    placeholder="optional"
                  />
                </div>
                <div className="field-schema-add-group">
                  <label>Max Range</label>
                  <input
                    type="number"
                    className="field-schema-add-input"
                    value={newFieldRangeMax}
                    onChange={(e) => setNewFieldRangeMax(e.target.value)}
                    placeholder="optional"
                  />
                </div>
              </div>
            )}

            <div className="field-schema-add-row">
              <label className="field-schema-checkbox">
                <input
                  type="checkbox"
                  checked={newFieldIsTarget}
                  onChange={(e) => setNewFieldIsTarget(e.target.checked)}
                />
                ML Target
              </label>
              <label className="field-schema-checkbox">
                <input
                  type="checkbox"
                  checked={newFieldIsComposite}
                  onChange={(e) => setNewFieldIsComposite(e.target.checked)}
                />
                Composite Only
              </label>
              <button
                className="elem-add-btn"
                onClick={handleAddField}
                disabled={!newFieldName.trim() || !!fieldNameError || addingField}
              >
                <Plus size={14} /> {addingField ? "Adding..." : "Add Field"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Field List */}
      {fieldSchemaLoading ? (
        <div className="settings-loading">Loading field schema...</div>
      ) : (
        <div className="field-schema-list">
          {/* Target fields */}
          {filteredFields.filter((f) => f.is_target).length > 0 && (
            <div className="field-schema-group">
              <h4 className="field-schema-group-title">
                <Layers size={14} /> Target Fields
              </h4>
              {filteredFields.filter((f) => f.is_target).map(renderFieldRow)}
            </div>
          )}

          {/* Categorical fields */}
          {categoricalFields.length > 0 && (
            <div className="field-schema-group">
              <h4 className="field-schema-group-title">
                <Tag size={14} /> Categorical Fields
              </h4>
              {categoricalFields.map(renderFieldRow)}
            </div>
          )}

          {/* Numeric fields */}
          {numericFields.filter((f) => !f.is_target).length > 0 && (
            <div className="field-schema-group">
              <h4 className="field-schema-group-title">
                <Hash size={14} /> Numeric Input Fields
              </h4>
              {numericFields.filter((f) => !f.is_target).map(renderFieldRow)}
            </div>
          )}

          {/* String/metadata fields */}
          {otherFields.length > 0 && (
            <div className="field-schema-group">
              <h4 className="field-schema-group-title">
                <Type size={14} /> Metadata Fields
              </h4>
              {otherFields.map(renderFieldRow)}
            </div>
          )}
        </div>
      )}

      {/* Info footer */}
      <div className="field-schema-footer">
        <Info size={11} />
        <span>
          Changes to the field schema take effect immediately for new operations.
          Existing trained models will continue to use the schema they were trained with.
        </span>
      </div>
    </div>
  );
}
