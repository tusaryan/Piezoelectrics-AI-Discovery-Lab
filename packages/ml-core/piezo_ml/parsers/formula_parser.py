import re
from typing import Dict
import chemparse
from .formula_normalizer import normalize_formula

class FormulaParseError(Exception):
    def __init__(self, message: str, position: int = None, char: str = None):
        super().__init__(message)
        self.position = position
        self.char = char

class FormulaParser:
    """
    Advanced parser for chemical formulas handling:
    - Standard perovskites: KNbO3
    - Solid solutions: (K0.5Na0.5)NbO3
    - Multi-phase mixtures: 0.96(K0.5Na0.5)NbO3-0.04Bi0.5Na0.5TiO3
    """
    
    @staticmethod
    def parse(formula: str) -> Dict[str, float]:
        if formula is None:
            raise FormulaParseError("Formula cannot be None")
            
        clean_formula = normalize_formula(formula)
        
        if not clean_formula:
            raise FormulaParseError("Empty formula provided")
        # Auto-fix common encoding artifacts from XLSX->CSV conversion
        # e.g., 'â€“' or 'âˆ’' or em-dash '—'
        # Full-width Unicode brackets (ï¼ˆ, ï¼‰, （, ）)
        clean_formula = (
            clean_formula
            .replace("â€“", "-")
            .replace("âˆ’", "-")
            .replace("—", "-")
            .replace("–", "-")
            .replace("−", "-")
            .replace("ï¼ˆ", "(")
            .replace("ï¼‰", ")")
            .replace("（", "(")
            .replace("）", ")")
            .replace(" ", "") # strip all internal spaces
        )
            
        # Validate exact characters (A-Z, a-z, 0-9, ., (, ), -, +, [, ])
        invalid_match = re.search(r'[^A-Za-z0-9.()\-\+\[\]]', clean_formula)
        if invalid_match:
            pos = invalid_match.start()
            char = invalid_match.group()
            raise FormulaParseError(f"Unknown character '{char}' at position {pos} in '{formula}'", pos, char)
            
        try:
            # Handle multi-phase formulas separated by dash at top level: A - B
            if FormulaParser._has_toplevel_separator(clean_formula):
                return FormulaParser._parse_multiphase(clean_formula)
            else:
                return FormulaParser._parse_single_phase(clean_formula)
        except Exception as e:
            if isinstance(e, FormulaParseError):
                raise e
            raise FormulaParseError(f"Failed to parse formula '{formula}': {str(e)}")

    @staticmethod
    def _has_toplevel_separator(formula: str) -> bool:
        """Check if formula has a '-' or '+' at parenthesis depth 0."""
        depth = 0
        for i, c in enumerate(formula):
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c in '-+' and depth == 0 and i > 0:
                # Make sure it's a phase separator, not part of a number
                # A phase separator is preceded by a letter/digit/closing bracket, not by nothing
                prev = formula[i-1] if i > 0 else ''
                if prev and (prev.isalpha() or prev.isdigit() or prev in ')]}'):
                    return True
        return False

    @staticmethod
    def _split_at_toplevel(formula: str) -> list:
        """Split formula on '-' or '+' only at parenthesis depth 0."""
        parts = []
        current = []
        depth = 0
        
        for i, c in enumerate(formula):
            if c in '([':
                depth += 1
                current.append(c)
            elif c in ')]':
                depth -= 1
                current.append(c)
            elif c in '-+' and depth == 0 and i > 0:
                prev = formula[i-1] if i > 0 else ''
                if prev and (prev.isalpha() or prev.isdigit() or prev in ')]}'):
                    parts.append(''.join(current).strip())
                    current = []
                else:
                    current.append(c)
            else:
                current.append(c)
        
        if current:
            parts.append(''.join(current).strip())
        
        return [p for p in parts if p]

    @staticmethod
    def _parse_single_phase(formula: str) -> Dict[str, float]:
        # Handle decimal limits gracefully
        parsed = chemparse.parse_formula(formula)
        
        # If it returns empty, it usually means bad bracket syntax
        if not parsed and formula:
            raise FormulaParseError(f"Failed to extract elements from formula. Check brackets and stoichiometry.")
            
        # Convert all to float
        result = {}
        for element, count in parsed.items():
            if count <= 0:
                raise FormulaParseError(f"Invalid stoichiometry {count} for element {element}")
            result[element] = float(count)
            
        return result

    @staticmethod
    def _parse_multiphase(formula: str) -> Dict[str, float]:
        """
        Parses multi-phase formulas like:
        - 0.96(K0.5Na0.5)NbO3-0.04CaZrO3
        - 0.95(0.995Na0.5K0.5NbO3-0.005BiFeO3)-0.05LiSbO3  (nested!)
        - (K0.44Na0.52)(Nb0.86Ta0.10)O3+0.04LiSbO3
        
        Splits ONLY at top-level dashes (respects parenthesis depth).
        """
        parts = FormulaParser._split_at_toplevel(formula)
        final_composition = {}
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Extract leading multiplier if exists
            match = re.match(r'^([\d.]+)(.*)$', part)
            if match:
                multiplier_str, rest_of_formula = match.groups()
                multiplier = float(multiplier_str)
            else:
                multiplier = 1.0
                rest_of_formula = part
            
            if not rest_of_formula:
                continue
                
            # Check if this part itself contains nested phases (e.g., inside parens)
            # If rest_of_formula is like (0.995Na0.5K0.5NbO3-0.005BiFeO3)
            if rest_of_formula.startswith('(') and rest_of_formula.endswith(')'):
                inner = rest_of_formula[1:-1]
                # Check if the inner formula is itself multi-phase
                if FormulaParser._has_toplevel_separator(inner):
                    parsed_part = FormulaParser._parse_multiphase(inner)
                else:
                    parsed_part = FormulaParser._parse_single_phase(rest_of_formula)
            elif rest_of_formula.startswith('[') and rest_of_formula.endswith(']'):
                inner = rest_of_formula[1:-1]
                if FormulaParser._has_toplevel_separator(inner):
                    parsed_part = FormulaParser._parse_multiphase(inner)
                else:
                    parsed_part = FormulaParser._parse_single_phase(rest_of_formula)
            else:
                parsed_part = FormulaParser._parse_single_phase(rest_of_formula)
                
            for el, count in parsed_part.items():
                final_composition[el] = final_composition.get(el, 0) + (count * multiplier)
                
        # Round to 4 decimal places to avoid floating point artifacts
        return {el: round(count, 4) for el, count in final_composition.items()}

