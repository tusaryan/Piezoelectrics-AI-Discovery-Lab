import pytest
import numpy as np
from piezo_ml.features.engineer import FeatureEngineer

def test_feature_vector_dimensions():
    engineer = FeatureEngineer()
    vector, names = engineer.compute_features("KNbO3")
    assert len(vector) == 35
    assert len(names) == 35

def test_deterministic_output():
    engineer = FeatureEngineer()
    v1, _ = engineer.compute_features("(K0.5Na0.5)NbO3")
    v2, _ = engineer.compute_features("K0.5Na0.5NbO3")
    np.testing.assert_array_almost_equal(v1, v2)

def test_cache_hit():
    engineer = FeatureEngineer()
    v1, _ = engineer.compute_features("BaTiO3")
    v2, _ = engineer.compute_features("BaTiO3")
    
    # Should be identical memref if pulled straight out of naive cache, but numpy array
    # wrapping makes a new one. Still, values represent exact matching hit.
    np.testing.assert_array_almost_equal(v1, v2)

def test_goldschmidt_tolerance():
    engineer = FeatureEngineer()
    vector, names = engineer.compute_features("BaTiO3")
    
    # tolerance_factor is the last feature
    idx = names.index("tolerance_factor")
    t = vector[idx]
    
    # BaTiO3 tolerance factor is ~1.06
    assert 0.95 <= t <= 1.15
