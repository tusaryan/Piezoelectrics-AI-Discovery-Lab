"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Atom, Plus, Trash2, Zap, RefreshCw, RotateCcw, X, UserPlus,
  Info, AlertCircle,
} from "lucide-react";
import { useSettingsStore } from "@/lib/store/settingsStore";

/**
 * PendingElements — Central element management UI.
 * Add new elements, manage properties, view registry, with undo and reset.
 */
export default function PendingElements() {
  const {
    elementRegistry, elementsLoading, fetchElements,
    addPendingElement, removePendingElement, bootstrapElements,
    removeSupportedElement, addCustomProperty, removeCustomProperty,
    resetElementsAndProperties,
  } = useSettingsStore();

  const [newSymbol, setNewSymbol] = useState("");
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: "success" | "error" } | null>(null);
  const [adding, setAdding] = useState(false);
  const [bootstrapping, setBootstrapping] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [showAllProps, setShowAllProps] = useState(false);
  const [newPropKey, setNewPropKey] = useState("");
  const [addingProp, setAddingProp] = useState(false);
  const [showAddProp, setShowAddProp] = useState(false);
  const [catError, setCatError] = useState("");

  // Undo state with countdown
  const [undoAction, setUndoAction] = useState<{ type: string; key: string; timer: NodeJS.Timeout } | null>(null);
  const [undoSecondsLeft, setUndoSecondsLeft] = useState(20);
  const undoIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Confirm dialogs
  const [confirmRemoveElem, setConfirmRemoveElem] = useState<string | null>(null);
  const [confirmRemoveProp, setConfirmRemoveProp] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);

  useEffect(() => { fetchElements(); }, [fetchElements]);

  // Cleanup undo timers on unmount
  useEffect(() => {
    return () => {
      if (undoAction?.timer) clearTimeout(undoAction.timer);
      if (undoIntervalRef.current) clearInterval(undoIntervalRef.current);
    };
  }, [undoAction]);

  const categories = ["A-site", "B-site", "dopant", "rare_earth"];

  const startUndoCountdown = (type: string, key: string) => {
    // Clear previous
    if (undoIntervalRef.current) clearInterval(undoIntervalRef.current);
    if (undoAction?.timer) clearTimeout(undoAction.timer);

    setUndoSecondsLeft(20);
    const mainTimer = setTimeout(() => {
      setUndoAction(null);
      setUndoSecondsLeft(0);
      if (undoIntervalRef.current) clearInterval(undoIntervalRef.current);
    }, 20000);

    const interval = setInterval(() => {
      setUndoSecondsLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    undoIntervalRef.current = interval;

    setUndoAction({ type, key, timer: mainTimer });
  };

  // Superheavy confirm state
  const [superheavyConfirm, setSuperheavyConfirm] = useState<string | null>(null);
  const [pendingAddData, setPendingAddData] = useState<{ symbol: string; cats: string[] } | null>(null);

  const doAdd = async (symbol: string, cats: string[]) => {
    setAdding(true);
    try {
      await addPendingElement(symbol, cats);
      setNewSymbol("");
      setSelectedCategories([]);
      setStatusMsg({ text: `${symbol} added to pending elements`, type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Failed to add element", type: "error" });
    }
    setAdding(false);
  };

  const handleAdd = async () => {
    const raw = newSymbol.trim();
    if (!raw) return;
    if (selectedCategories.length === 0) {
      setCatError("Please select at least one category (A-site, B-site, dopant, or rare_earth)");
      return;
    }
    // Auto-capitalize: first char upper, rest lower
    const symbol = raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
    // Validate format
    if (!/^[A-Z][a-z]{0,2}$/.test(symbol)) {
      setCatError("Symbol must be 1-3 characters: first uppercase, rest lowercase (e.g., Na, Ce, Uue)");
      return;
    }
    setCatError("");
    // Superheavy element confirmation (3-char symbols like Uue, Uun)
    if (symbol.length === 3) {
      setSuperheavyConfirm(symbol);
      setPendingAddData({ symbol, cats: [...selectedCategories] });
      return;
    }
    await doAdd(symbol, selectedCategories);
  };

  const confirmSuperheavy = async () => {
    if (pendingAddData) {
      await doAdd(pendingAddData.symbol, pendingAddData.cats);
    }
    setSuperheavyConfirm(null);
    setPendingAddData(null);
  };

  const handleBootstrap = async () => {
    setBootstrapping(true);
    try {
      const msg = await bootstrapElements();
      setStatusMsg({ text: msg, type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Bootstrap failed", type: "error" });
    }
    setBootstrapping(false);
  };

  const toggleCategory = (cat: string) => {
    setCatError("");
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
    );
  };

  const handleRemoveElement = async (symbol: string) => {
    try {
      await removeSupportedElement(symbol);
      setConfirmRemoveElem(null);
      startUndoCountdown("element", symbol);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Cannot remove default element", type: "error" });
      setConfirmRemoveElem(null);
    }
  };

  const handleRemoveProp = async (key: string) => {
    try {
      await removeCustomProperty(key);
      setConfirmRemoveProp(null);
      startUndoCountdown("property", key);
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Cannot remove default property", type: "error" });
      setConfirmRemoveProp(null);
    }
  };

  const handleUndo = async () => {
    if (!undoAction) return;
    if (undoAction.type === "element") {
      try {
        await addPendingElement(undoAction.key, []);
        setStatusMsg({ text: `${undoAction.key} re-added to pending. Bootstrap to restore.`, type: "success" });
      } catch {}
    } else if (undoAction.type === "property") {
      try {
        await addCustomProperty(undoAction.key);
        setStatusMsg({ text: `Property '${undoAction.key}' restored.`, type: "success" });
      } catch {}
    }
    clearTimeout(undoAction.timer);
    if (undoIntervalRef.current) clearInterval(undoIntervalRef.current);
    setUndoAction(null);
    setUndoSecondsLeft(0);
  };

  const handleAddProp = async () => {
    if (!newPropKey.trim()) return;
    setAddingProp(true);
    try {
      await addCustomProperty(newPropKey.trim().toLowerCase().replace(/\s+/g, "_"));
      setNewPropKey("");
      setShowAddProp(false);
      setStatusMsg({ text: `Property added successfully`, type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Failed to add property", type: "error" });
    }
    setAddingProp(false);
  };

  const handleReset = async () => {
    setResetting(true);
    try {
      const result = await resetElementsAndProperties();
      setStatusMsg({ text: result.message, type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: e.message || "Reset failed", type: "error" });
    }
    setResetting(false);
    setConfirmReset(false);
  };

  const pending = elementRegistry?.pending_elements || [];
  const supported = elementRegistry?.supported_elements || [];
  const userAddedElems = new Set(elementRegistry?.user_added_elements || []);
  const userAddedProps = new Set(elementRegistry?.user_added_properties || []);
  const displayElements = showAll ? supported : supported.slice(0, 12);
  const allProps = elementRegistry?.property_keys || [];
  const displayProps = showAllProps ? allProps : allProps.slice(0, 8);

  const categoryColors: Record<string, string> = {
    "A-site": "#6366F1",
    "B-site": "#10B981",
    "dopant": "#F59E0B",
    "rare_earth": "#8B5CF6",
    "anion": "#EF4444",
    "other": "#6B7280",
  };

  return (
    <div className="settings-section">
      <div className="settings-section-header">
        <div className="settings-section-icon"><Atom size={18} /></div>
        <div>
          <h3>Element Registry</h3>
          <p className="settings-section-desc">
            {supported.length} elements supported · {elementRegistry?.total_properties || 0} properties each
          </p>
        </div>
        <div className="settings-section-header-actions">
          {(userAddedElems.size > 0 || userAddedProps.size > 0) && (
            confirmReset ? (
              <div className="elem-reset-confirm">
                <span>Reset all to defaults?</span>
                <button className="model-lib-btn danger" onClick={handleReset} disabled={resetting}>
                  {resetting ? "Resetting..." : "Yes"}
                </button>
                <button className="model-lib-btn" onClick={() => setConfirmReset(false)}>No</button>
              </div>
            ) : (
              <button className="settings-reset-btn-text" onClick={() => setConfirmReset(true)}
                title="Reset elements and properties to defaults">
                <RotateCcw size={13} /> Reset
              </button>
            )
          )}
          <button className="settings-refresh-btn" onClick={fetchElements} title="Refresh">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Undo banner with countdown */}
      {undoAction && (
        <div className="elem-undo-banner">
          <span>
            {undoAction.type === "element" ? `Element "${undoAction.key}" removed.` : `Property "${undoAction.key}" removed.`}
          </span>
          <button className="elem-undo-btn" onClick={handleUndo}>
            <RotateCcw size={12} /> Undo ({undoSecondsLeft}s)
          </button>
        </div>
      )}

      {/* Status message */}
      {statusMsg && (
        <div className={`elem-status-msg ${statusMsg.type}`}>
          {statusMsg.type === "error" && <AlertCircle size={13} />}
          {statusMsg.text}
          <button onClick={() => setStatusMsg(null)} className="danger-result-close">×</button>
        </div>
      )}

      {/* Add New Element */}
      <div className="elem-add-section">
        <h4>Add New Element</h4>
        <div className="elem-add-row">
          <input
            className="elem-add-input"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value)}
            placeholder="Symbol (e.g., Ce, Y)"
            maxLength={3}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
          />
          <div className="elem-cat-pills">
            {categories.map((cat) => (
              <button
                key={cat}
                className={`elem-cat-pill ${selectedCategories.includes(cat) ? "selected" : ""}`}
                onClick={() => toggleCategory(cat)}
                style={selectedCategories.includes(cat) ? { background: `${categoryColors[cat]}22`, color: categoryColors[cat], borderColor: categoryColors[cat] } : {}}
              >
                {cat}
              </button>
            ))}
          </div>
          <button
            className="elem-add-btn"
            onClick={handleAdd}
            disabled={!newSymbol.trim() || adding}
          >
            <Plus size={14} /> {adding ? "Adding..." : "Add"}
          </button>
        </div>
        {catError && (
          <div className="elem-cat-error">
            <AlertCircle size={12} /> {catError}
          </div>
        )}
        {superheavyConfirm && (
          <div className="elem-superheavy-confirm">
            <AlertCircle size={13} />
            <span>
              <strong>{superheavyConfirm}</strong> is a 3-character symbol (typically used for superheavy/transactinide elements like Uue, Uun).
              Are you sure you want to add this element?
            </span>
            <div className="elem-superheavy-actions">
              <button className="elem-add-btn small" onClick={confirmSuperheavy} disabled={adding}>
                {adding ? "Adding..." : "Yes, Add"}
              </button>
              <button className="model-lib-btn" onClick={() => { setSuperheavyConfirm(null); setPendingAddData(null); }}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Pending Elements */}
      {pending.length > 0 && (
        <div className="elem-pending-section">
          <div className="elem-pending-header">
            <h4>Pending Elements ({pending.length})</h4>
            <button
              className="elem-bootstrap-btn"
              onClick={handleBootstrap}
              disabled={bootstrapping}
            >
              <Zap size={14} />
              {bootstrapping ? "Bootstrapping..." : "Bootstrap All"}
            </button>
          </div>
          <div className="elem-pending-list">
            {pending.map((sym) => (
              <div key={sym} className="elem-pending-item">
                <span className="elem-symbol">{sym}</span>
                <button
                  className="elem-remove-btn"
                  onClick={() => removePendingElement(sym)}
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Supported Elements Grid */}
      <div className="elem-registry-section">
        <h4>Supported Elements</h4>
        {elementsLoading ? (
          <div className="settings-loading">Loading registry...</div>
        ) : (
          <>
            <div className="elem-grid">
              {displayElements.map((el) => (
                <div key={el.symbol}
                  className={`elem-card ${el.is_user_added ? "user-added" : ""}`}
                  title={`${el.symbol} — ${el.category}`}
                >
                  <span className="elem-number">{el.atomic_number}</span>
                  <span className="elem-symbol-lg">{el.symbol}</span>
                  <div className="elem-cat-badge-wrap"
                    onMouseEnter={() => el.category === "other" ? setHoveredCategory(el.symbol) : null}
                    onMouseLeave={() => setHoveredCategory(null)}
                  >
                    <span
                      className="elem-cat-badge"
                      style={{ background: `${categoryColors[el.category] || "#6B7280"}22`, color: categoryColors[el.category] || "#6B7280" }}
                    >
                      {el.category}
                    </span>
                    {el.category === "other" && hoveredCategory === el.symbol && (
                      <div className="elem-cat-tooltip">
                        Not classified into A-site, B-site, dopant, or rare_earth
                      </div>
                    )}
                  </div>
                  {el.is_user_added && (
                    <span className="elem-user-badge" title="User-added element">
                      <UserPlus size={9} />
                    </span>
                  )}
                  {el.is_user_added && (
                    confirmRemoveElem === el.symbol ? (
                      <div className="elem-card-confirm">
                        <button className="elem-card-confirm-yes" onClick={() => handleRemoveElement(el.symbol)}>✓</button>
                        <button className="elem-card-confirm-no" onClick={() => setConfirmRemoveElem(null)}>✗</button>
                      </div>
                    ) : (
                      <button className="elem-card-remove" onClick={() => setConfirmRemoveElem(el.symbol)}
                        title="Remove user-added element">
                        <X size={10} />
                      </button>
                    )
                  )}
                </div>
              ))}
            </div>
            {supported.length > 12 && (
              <button className="elem-show-all" onClick={() => setShowAll(!showAll)}>
                {showAll ? "Show Less" : `Show All ${supported.length} Elements`}
              </button>
            )}
          </>
        )}
      </div>

      {/* Properties Info */}
      {elementRegistry && (
        <div className="elem-props-info">
          <div className="elem-props-header">
            <strong>{elementRegistry.total_properties} properties</strong> per element:
            <div className="elem-props-actions">
              <button className="elem-add-prop-toggle" onClick={() => setShowAddProp(!showAddProp)}>
                <Plus size={12} /> Add Property
              </button>
            </div>
          </div>

          {showAddProp && (
            <div className="elem-add-prop-row">
              <input
                className="elem-add-prop-input"
                value={newPropKey}
                onChange={(e) => setNewPropKey(e.target.value)}
                placeholder="snake_case name (e.g., dielectric_constant)"
                onKeyDown={(e) => e.key === "Enter" && handleAddProp()}
              />
              <button className="elem-add-btn small" onClick={handleAddProp} disabled={!newPropKey.trim() || addingProp}>
                <Plus size={12} /> {addingProp ? "Adding..." : "Add"}
              </button>
              <div className="elem-add-prop-info">
                <Info size={11} />
                <span>New properties will have null values for all elements until populated. A server restart may be needed for full training pipeline integration.</span>
              </div>
            </div>
          )}

          <div className="elem-props-list">
            {displayProps.map((k) => (
              <span key={k} className={`elem-prop-tag ${userAddedProps.has(k) ? "user-added" : ""}`}>
                {k}
                {userAddedProps.has(k) && (
                  <span className="elem-prop-user-icon" title="User-added property">
                    <UserPlus size={8} />
                  </span>
                )}
                {userAddedProps.has(k) && (
                  confirmRemoveProp === k ? (
                    <span className="elem-prop-confirm">
                      <button onClick={() => handleRemoveProp(k)} title="Confirm remove">✓</button>
                      <button onClick={() => setConfirmRemoveProp(null)} title="Cancel">✗</button>
                    </span>
                  ) : (
                    <button className="elem-prop-remove" onClick={() => setConfirmRemoveProp(k)}
                      title="Remove property">
                      <X size={8} />
                    </button>
                  )
                )}
              </span>
            ))}
            {allProps.length > 8 && (
              <button
                className="elem-prop-tag more clickable"
                onClick={() => setShowAllProps(!showAllProps)}
              >
                {showAllProps ? "Show Less" : `+${allProps.length - 8} more`}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
