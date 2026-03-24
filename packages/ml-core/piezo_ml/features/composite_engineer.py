import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from piezo_ml.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

# Encodings for categoricals from PIEZO_AI_SPEC 13.1
MATRIX_MAP = {"pvdf": 0, "pvdf_trfe": 1, "pzt": 2, "epoxy": 3, "silicone": 4, "unknown": -1}
MORPHOLOGY_MAP = {"spherical": 0, "rod": 1, "fiber": 2, "sheet": 3, "unknown": -1}
TREATMENT_MAP = {"none": 0, "dopamine": 1, "silane": 2, "fluorinated": 3, "unknown": -1}
FABRICATION_MAP = {"solvent_cast": 0, "electrospin": 1, "hot_press": 2, "3d_print": 3, "unknown": -1}

class CompositeFeatureEngineer(FeatureEngineer):
    """
    Extends base FeatureEngineer (35 dims) to 43 dimensions for composite materials.
    Adds 8 specific features:
    1. filler_wt_pct
    2. filler_vol_pct
    3. particle_size_nm
    4. beta_phase_pct (for PVDF matrix targets)
    5. matrix_type_encoded
    6. morphology_encoded
    7. treatment_encoded
    8. fabrication_encoded
    """

    def __init__(self):
        super().__init__()
        self.composite_feature_names = [
            "filler_wt_pct", "filler_vol_pct", "particle_size_nm", "beta_phase_pct",
            "matrix_type_enc", "morphology_enc", "treatment_enc", "fabrication_enc"
        ]
        
    def transform(self, element_fractions: Dict[str, float], composite_props: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Produce 43-dim vector: 35 base features + 8 composite features.
        If composite_props is None, safely falls back via zero-padding.
        """
        # Get base 35 features
        base_vector, base_names = super().transform(element_fractions)
        
        # Build 8 extra composite features
        comp_vec = np.zeros(8, dtype=np.float32)
        
        if composite_props:
            comp_vec[0] = float(composite_props.get("filler_wt_pct", 0.0) or 0.0)
            comp_vec[1] = float(composite_props.get("filler_vol_pct", 0.0) or 0.0)
            comp_vec[2] = float(composite_props.get("particle_size_nm", 0.0) or 0.0)
            comp_vec[3] = float(composite_props.get("beta_phase_pct", 0.0) or 0.0)
            
            mat_type = str(composite_props.get("matrix_type", "")).lower()
            comp_vec[4] = float(MATRIX_MAP.get(mat_type, MATRIX_MAP["unknown"]))
            
            morph = str(composite_props.get("particle_morphology", "")).lower()
            comp_vec[5] = float(MORPHOLOGY_MAP.get(morph, MORPHOLOGY_MAP["unknown"]))
            
            treat = str(composite_props.get("surface_treatment", "")).lower()
            comp_vec[6] = float(TREATMENT_MAP.get(treat, TREATMENT_MAP["unknown"]))
            
            fab = str(composite_props.get("fabrication_method", "")).lower()
            comp_vec[7] = float(FABRICATION_MAP.get(fab, FABRICATION_MAP["unknown"]))
        else:
            logger.warning("Composite Feature Engineering requested but no composite properties provided. Falling back to zero-padded base features.")
            
        full_vector = np.concatenate([base_vector, comp_vec])
        full_names = base_names + self.composite_feature_names
        
        return full_vector, full_names
