import re

# Dictionary to map subscript unicode characters to their regular integers
SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '▷': '7', '₈': '8', '₉': '9',
    # Adding some common accidental characters
    'O3': 'O3', 'o3': 'O3' 
}

def normalize_formula(formula: str) -> str:
    """
    Normalizes a chemical formula string by:
    1. Trimming whitespace
    2. Converting unicode subscripts to standard integers
    3. Fixing common typos (like 'o' instead of 'O' for Oxygen if it's clearly a typo, though careful)
    """
    if not formula:
        return ""
        
    formula = formula.strip()
    
    # Replace unicode subscripts
    for sub, normal in SUBSCRIPT_MAP.items():
        formula = formula.replace(sub, normal)
        
    # Remove all whitespace inside the formula
    formula = re.sub(r'\s+', '', formula)
    
    return formula
