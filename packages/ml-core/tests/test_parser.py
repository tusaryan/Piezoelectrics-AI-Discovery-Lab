import pytest
from piezo_ml.parsers.formula_parser import FormulaParser, FormulaParseError
from piezo_ml.parsers.formula_validator import FormulaValidator
from piezo_ml.parsers.formula_normalizer import normalize_formula

def test_normalization():
    assert normalize_formula("K N b O 3") == "KNbO3"
    assert normalize_formula("K0.5Na0.5NbO₃") == "K0.5Na0.5NbO3"
    assert normalize_formula(" Ba Ti O ₃ ") == "BaTiO3"

def test_simple_parser():
    parsed = FormulaParser.parse("KNbO3")
    assert parsed == {"K": 1.0, "Nb": 1.0, "O": 3.0}

def test_decimal_parser():
    parsed = FormulaParser.parse("K0.5Na0.5NbO3")
    assert parsed == {"K": 0.5, "Na": 0.5, "Nb": 1.0, "O": 3.0}

def test_nested_brackets():
    parsed = FormulaParser.parse("(K0.5Na0.5)NbO3")
    assert parsed == {"K": 0.5, "Na": 0.5, "Nb": 1.0, "O": 3.0}

def test_unicode_subscripts():
    parsed = FormulaParser.parse("K0.5Na0.5NbO₃")
    assert parsed == {"K": 0.5, "Na": 0.5, "Nb": 1.0, "O": 3.0}

def test_multi_phase():
    # 96% KNN, 4% BNT
    parsed = FormulaParser.parse("0.96(K0.5Na0.5NbO3)-0.04(Bi0.5Na0.5TiO3)")
    # K: 0.96 * 0.5 = 0.48
    # Na: (0.96 * 0.5) + (0.04 * 0.5) = 0.48 + 0.02 = 0.50
    # Nb: 0.96 * 1.0 = 0.96
    # O: (0.96 * 3) + (0.04 * 3) = 2.88 + 0.12 = 3.0
    # Bi: 0.04 * 0.5 = 0.02
    # Ti: 0.04 * 1.0 = 0.04
    assert parsed["K"] == 0.48
    assert parsed["Na"] == 0.50
    assert parsed["Nb"] == 0.96
    assert parsed["O"] == 3.0
    assert parsed["Bi"] == 0.02
    assert parsed["Ti"] == 0.04

def test_invalid_characters():
    with pytest.raises(FormulaParseError) as exc:
        FormulaParser.parse("K0.5NbO?")
    assert "?'" in str(exc.value)

def test_empty_formula():
    with pytest.raises(FormulaParseError):
        FormulaParser.parse("")
        
def test_none_formula():
    with pytest.raises(FormulaParseError):
        FormulaParser.parse(None)

def test_validator():
    parsed = FormulaParser.parse("K0.5Na0.5NbO2")
    warnings = FormulaValidator.validate_stoichiometry(parsed)
    # Shouldn't error, but should return a warning about O sum (though we only check A/B sums right now)
    assert len(warnings) == 0 # We left out explicit O-sum check, but A/B are 1.0

    parsed2 = FormulaParser.parse("K0.2Na0.2NbO3") # A-site sum = 0.4
    warnings2 = FormulaValidator.validate_stoichiometry(parsed2)
    assert any("A-site fractions sum" in w for w in warnings2)

    parsed3 = FormulaParser.parse("PbTiO3")
    warnings3 = FormulaValidator.validate_stoichiometry(parsed3)
    assert any("Lead (Pb)" in w for w in warnings3)
