"use client";

/**
 * CompositeFields — Matrix type, filler wt%, morphology, etc.
 */

import { FlaskConical } from "lucide-react";
import { usePredictStore } from "@/lib/store/predictStore";

const MATRIX_TYPES = [
  "none", "pvdf", "p_vdf_trfe", "pvdf_hfp", "pvdf_hfp_ctrfe",
  "epoxy", "silicone",
];

const MORPHOLOGIES = [
  "none", "spherical", "rod", "cube", "nanoblock", "fiber", "platelet",
];

const TREATMENTS = [
  "none", "untreated", "silane", "plasma", "acid", "peg", "dopamine",
];

const FABRICATION_METHODS = [
  "conventional", "hot_press", "sps", "electrospinning",
  "solvent_cast", "hot_compression",
];

export default function CompositeFields() {
  const {
    isComposite,
    setIsComposite,
    compositeParams,
    setCompositeParam,
  } = usePredictStore();

  return (
    <div className="predict-card">
      <div className="predict-card-title">
        <FlaskConical size={16} /> Composite Material
      </div>

      <div
        className="composite-toggle"
        onClick={() => setIsComposite(!isComposite)}
      >
        <div
          className={`composite-toggle-switch ${isComposite ? "active" : ""}`}
        />
        <span className="composite-toggle-label">
          {isComposite
            ? "Composite mode — polymer + ceramic filler"
            : "Bulk ceramic — no composite fields"}
        </span>
      </div>

      {isComposite && (
        <div className="composite-fields-grid">
          <div className="composite-field">
            <label htmlFor="matrix-type">Matrix Type</label>
            <select
              id="matrix-type"
              value={compositeParams.matrix_type || "none"}
              onChange={(e) => setCompositeParam("matrix_type", e.target.value)}
            >
              {MATRIX_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t === "none" ? "Select..." : t.replace(/_/g, "-").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          <div className="composite-field">
            <label htmlFor="filler-wt">Filler Wt%</label>
            <input
              id="filler-wt"
              type="number"
              min={0}
              max={100}
              step={1}
              value={compositeParams.filler_wt_pct ?? ""}
              onChange={(e) =>
                setCompositeParam(
                  "filler_wt_pct",
                  e.target.value ? parseFloat(e.target.value) : undefined,
                )
              }
              placeholder="e.g. 20"
            />
          </div>

          <div className="composite-field">
            <label htmlFor="morphology">Particle Morphology</label>
            <select
              id="morphology"
              value={compositeParams.particle_morphology || "none"}
              onChange={(e) =>
                setCompositeParam("particle_morphology", e.target.value)
              }
            >
              {MORPHOLOGIES.map((m) => (
                <option key={m} value={m}>
                  {m === "none" ? "Select..." : m.charAt(0).toUpperCase() + m.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="composite-field">
            <label htmlFor="particle-size">Particle Size (nm)</label>
            <input
              id="particle-size"
              type="number"
              min={0}
              step={10}
              value={compositeParams.particle_size_nm ?? ""}
              onChange={(e) =>
                setCompositeParam(
                  "particle_size_nm",
                  e.target.value ? parseFloat(e.target.value) : undefined,
                )
              }
              placeholder="e.g. 200"
            />
          </div>

          <div className="composite-field">
            <label htmlFor="treatment">Surface Treatment</label>
            <select
              id="treatment"
              value={compositeParams.surface_treatment || "none"}
              onChange={(e) =>
                setCompositeParam("surface_treatment", e.target.value)
              }
            >
              {TREATMENTS.map((t) => (
                <option key={t} value={t}>
                  {t === "none" ? "Select..." : t.charAt(0).toUpperCase() + t.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <div className="composite-field">
            <label htmlFor="fab-method">Fabrication Method</label>
            <select
              id="fab-method"
              value={compositeParams.fabrication_method || "conventional"}
              onChange={(e) =>
                setCompositeParam("fabrication_method", e.target.value)
              }
            >
              {FABRICATION_METHODS.map((m) => (
                <option key={m} value={m}>
                  {m.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
