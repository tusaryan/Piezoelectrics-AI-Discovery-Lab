from piezo_ml.parsers import FormulaParser


def test_parse_knn_formula():
    parser = FormulaParser()
    result = parser.parse("K0.5Na0.5NbO3")
    assert result.is_valid
    assert result.elements["K"] == 0.5
    assert result.elements["Na"] == 0.5
    assert result.elements["Nb"] == 1.0
    assert result.elements["O"] == 3.0


def test_parse_multiphase_formula():
    parser = FormulaParser()
    result = parser.parse("0.96(K0.48Na0.52)(Nb0.95Sb0.05)O3-0.04Bi0.5Na0.5ZrO3")
    assert result.is_valid
    assert abs(result.elements["K"] - 0.4608) < 1e-9
    assert abs(result.elements["Na"] - 0.5192) < 1e-9
