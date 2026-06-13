import pandas as pd

from piezo_ml.features import FeatureEngineer


def test_feature_vector_for_simple_formula():
    frame = pd.DataFrame(
        [
            {"uid": 1, "formula": "K0.5Na0.5NbO3"},
        ]
    )
    engineer = FeatureEngineer()
    vectors, parsed = engineer.engineer_dataframe(frame)
    assert len(vectors) == 1
    assert len(parsed) == 1
    assert abs(vectors.iloc[0]["frac_K"] - 0.25) < 1e-9
    assert abs(vectors.iloc[0]["frac_Na"] - 0.25) < 1e-9
    assert abs(vectors.iloc[0]["frac_Nb"] - 0.5) < 1e-9
    assert 0.0 < vectors.iloc[0]["tolerance_factor"] < 2.0
    assert 0.0 < vectors.iloc[0]["octahedral_factor"] < 2.0
