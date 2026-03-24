from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
from piezo_ml.parsers.formula_parser import FormulaParser, FormulaParseError

@dataclass
class DataIssue:
    row_idx: int
    column: str
    issue_type: str
    severity: str # 'critical', 'warning', 'info'
    description: str
    auto_fixable: bool
    choices: List[str]

class DataValidator:
    """
    Validates Piezo datasets without mutating them.
    Emits specific DataIssue warnings based on Spec 11.2 constraints.
    """
    
    # Unicode chars that appear in scientific formulas and their ASCII equivalents
    # ONLY includes chars that are genuinely different from ASCII — no '.' (that's a regular period)
    UNICODE_FIXES = {
        # Subscript digits
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        # Superscript digits
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
        # Dash variants (en-dash, em-dash, minus sign)
        '–': '-', '—': '-', '−': '-',
        # Dot variants
        '·': '.',
        # Fullwidth parentheses/brackets
        '（': '(', '）': ')',
        '［': '[', '］': ']',
    }
    
    @classmethod
    def normalize_formula(cls, formula: str) -> str:
        """Apply all Unicode fixes and strip whitespace."""
        result = formula
        for old, new in cls.UNICODE_FIXES.items():
            result = result.replace(old, new)
        return result.strip()
    
    @classmethod
    def validate_df(cls, df: pd.DataFrame) -> List[DataIssue]:
        issues = []
        
        # Ensure mandatory columns exist
        if 'formula' not in df.columns:
            issues.append(DataIssue(-1, 'global', 'missing_column', 'critical', 
                                  "Dataset missing required 'formula' column. Please go back and map your columns.",
                                  False, ["Go Back to Mapping"]))
            return issues
            
        has_d33 = 'd33' in df.columns
        has_tc = 'tc' in df.columns
        
        if not has_d33 and not has_tc:
            issues.append(DataIssue(-1, 'global', 'missing_target', 'warning',
                                  "Dataset has neither 'd33' nor 'tc' columns. You can still proceed, but ML training will be limited.",
                                  False, ["Proceed Anyway", "Go Back to Mapping"]))
            
        for idx, row in df.iterrows():
            formula = str(row.get('formula', ''))
            
            if not formula or formula.lower() == 'nan':
                issues.append(DataIssue(idx, 'formula', 'missing', 'critical',
                                      "Empty chemical formula", False, ["Edit Manually", "Drop Row"]))
                continue
            
            # Normalize first — fix Unicode chars, strip whitespace
            normalized = cls.normalize_formula(formula)
            
            # Only report Unicode issue if the formula ACTUALLY CHANGED
            if normalized != formula:
                issues.append(DataIssue(idx, 'formula', 'unicode', 'info',
                                      f"Contains special characters: '{formula.strip()}' → '{normalized}'",
                                      True, ["Auto-Fix", "Edit Manually", "Keep As-Is"]))
            
            # Parse the NORMALIZED formula — don't parse the raw one with bad chars
            try:
                FormulaParser.parse(normalized)
            except FormulaParseError as e:
                issues.append(DataIssue(idx, 'formula', 'parse_error', 'critical',
                                      f"Cannot parse formula '{normalized}': {str(e)}",
                                      False, ["Edit Manually", "Drop Row"]))
                
            # Validate targets
            if has_d33:
                d33_val = row['d33']
                if pd.isna(d33_val) or str(d33_val).strip() in ['', 'nan', 'NaN']:
                    issues.append(DataIssue(idx, 'd33', 'missing_target', 'warning',
                                          "Missing d33 value", True, ["KNN Impute", "Keep Empty", "Drop Row"]))
                else:
                    try:
                        f_val = float(d33_val)
                        if f_val < 0 or f_val > 5000:
                            issues.append(DataIssue(idx, 'd33', 'out_of_bounds', 'warning',
                                                  f"d33 = {d33_val} is outside expected range (0–5000 pC/N)",
                                                  False, ["Keep (suspicious)", "Edit Manually", "Drop Row"]))
                    except ValueError:
                        issues.append(DataIssue(idx, 'd33', 'invalid_number', 'warning',
                                              f"d33 = '{d33_val}' is not a valid number",
                                              False, ["Edit Manually", "Drop Row"]))
                                          
            if has_tc:
                tc_val = row['tc']
                if pd.isna(tc_val) or str(tc_val).strip() in ['', 'nan', 'NaN']:
                    issues.append(DataIssue(idx, 'tc', 'missing_target', 'warning',
                                          "Missing Tc value", True, ["KNN Impute", "Keep Empty", "Drop Row"]))
                else:
                    try:
                        f_val = float(tc_val)
                        if f_val < -273 or f_val > 2000:
                            issues.append(DataIssue(idx, 'tc', 'out_of_bounds', 'warning',
                                                  f"Tc = {tc_val} is outside expected range (−273–2000 °C)",
                                                  False, ["Keep (suspicious)", "Edit Manually", "Drop Row"]))
                    except ValueError:
                        issues.append(DataIssue(idx, 'tc', 'invalid_number', 'warning',
                                              f"Tc = '{tc_val}' is not a valid number",
                                              False, ["Edit Manually", "Drop Row"]))
                                          
        return issues

