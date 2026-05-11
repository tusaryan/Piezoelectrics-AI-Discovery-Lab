"use client";

/**
 * Interpretability Page — SHAP analysis, Physics Validation, Symbolic Regression.
 */

import { useEffect } from "react";
import { Eye, RefreshCw, Loader2 } from "lucide-react";
import { useInterpretStore } from "@/lib/store/interpretStore";
import InterpretModelSelector from "@/components/interpret/ModelSelector";
import ShapBeeswarm from "@/components/interpret/ShapBeeswarm";
import ShapWaterfall from "@/components/interpret/ShapWaterfall";
import ShapDependence from "@/components/interpret/ShapDependence";
import PhysicsValidation from "@/components/interpret/PhysicsValidation";
import SymbolicRegression from "@/components/interpret/SymbolicRegression";

export default function InterpretPage() {
  const {
    selectedModelId,
    beeswarmLoading,
    waterfallLoading,
    physicsLoading,
    loadModels,
    fetchBeeswarm,
    fetchWaterfall,
    fetchPhysicsValidation,
  } = useInterpretStore();

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleRefresh = () => {
    if (!selectedModelId) return;
    fetchBeeswarm();
    fetchWaterfall(0);
    fetchPhysicsValidation();
  };

  const isLoading = beeswarmLoading || waterfallLoading || physicsLoading;

  return (
    <div className="page-container interpret-page">
      {/* Header */}
      <div className="page-header">
        <div className="page-header-icon">
          <Eye size={22} />
        </div>
        <div className="page-header-text">
          <h1>Interpretability</h1>
          <p>Understand why your model makes predictions — SHAP analysis, physics validation, and symbolic regression</p>
        </div>
        <div className="page-header-actions">
          <button
            className="refresh-btn"
            onClick={handleRefresh}
            disabled={!selectedModelId || isLoading}
            title="Re-run all analyses"
          >
            {isLoading ? <Loader2 size={16} className="spin" /> : <RefreshCw size={16} />}
          </button>
        </div>
      </div>

      {/* Model Selector */}
      <InterpretModelSelector />

      {/* Analysis Section — only shown when model is selected */}
      {selectedModelId && (
        <div className="interpret-grid">
          {/* Row 1: Beeswarm (full width) */}
          <div className="interpret-row interpret-row-full">
            <ShapBeeswarm />
          </div>

          {/* Row 2: Waterfall + Dependence */}
          <div className="interpret-row interpret-row-half">
            <ShapWaterfall />
            <ShapDependence />
          </div>

          {/* Row 3: Physics Validation + Symbolic Regression */}
          <div className="interpret-row interpret-row-half">
            <PhysicsValidation />
            <SymbolicRegression />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!selectedModelId && (
        <div className="interpret-empty-page">
          <Eye size={40} className="empty-icon" />
          <h3>Select a Model to Analyze</h3>
          <p>
            Choose a trained model above to generate SHAP analysis,
            physics validation, and symbolic regression results.
          </p>
        </div>
      )}
    </div>
  );
}
