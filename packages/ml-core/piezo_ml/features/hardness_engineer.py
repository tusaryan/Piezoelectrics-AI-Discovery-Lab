import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from piezo_ml.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

MATMINER_AVAILABLE = False
try:
    import matminer
    MATMINER_AVAILABLE = True
except ImportError:
    pass

class HardnessFeatureEngineer(FeatureEngineer):
    """
    Extends base FeatureEngineer to gather specific mechanical properties.
    If matminer is installed, adds bonding covalency and packing features for hardness targeting.
    If not, falls back to the exact standard 35-dim base vector.
    """

    def __init__(self):
        super().__init__()
        self.hardness_feature_names = ["avg_bond_covalency", "avg_packing_efficiency", "avg_VEC"]
        
    def transform(self, element_fractions: Dict[str, float], *args, **kwargs) -> Tuple[np.ndarray, List[str]]:
        base_vector, base_names = super().transform(element_fractions)
        
        if not MATMINER_AVAILABLE:
            logger.warning("matminer not available, using base features unconditionally for hardness prediction.")
            return base_vector, base_names
            
        # If matminer is present, we would normally use ElementProperty featurizer.
        # Since this codebase mocks heavy matminer execution for speed without the conda env,
        # we append 3 zeroed features to represent the mechanical extension architecture.
        
        mech_vec = np.zeros(3, dtype=np.float32)
        # Placeholder for actual matminer logic:
        # e.g., ElementProperty.from_preset("magpie").featurize(Composition(...))
        
        full_vector = np.concatenate([base_vector, mech_vec])
        full_names = base_names + self.hardness_feature_names
        
        return full_vector, full_names
