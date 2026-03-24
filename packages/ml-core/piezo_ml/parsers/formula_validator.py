from typing import Dict, List

class FormulaValidator:
    # A-site elements common in piezoelectrics
    A_SITE_ELEMENTS = {'K', 'Na', 'Li', 'Bi', 'Ba', 'Ca', 'Sr', 'Pb', 'Ag'}
    # B-site elements common in piezoelectrics
    B_SITE_ELEMENTS = {'Nb', 'Ta', 'Sb', 'Ti', 'Zr', 'Hf', 'W', 'Mo', 'Sn', 'Sc'}
    
    @classmethod
    def validate_stoichiometry(cls, parsed_dict: Dict[str, float]) -> List[str]:
        """
        Validates the parsed dictionary to ensure atomic fractions make physical sense
        Returns a list of warning messages (empty list if perfectly valid).
        """
        warnings = []
        
        if not parsed_dict:
            return ["Empty formula parsed"]
            
        if 'O' not in parsed_dict:
            warnings.append("No Oxygen (O) detected in perovskite formula")
            
        a_site_sum = sum(parsed_dict.get(el, 0) for el in cls.A_SITE_ELEMENTS)
        b_site_sum = sum(parsed_dict.get(el, 0) for el in cls.B_SITE_ELEMENTS)
        
        # In a standard ABO3 perovskite, A and B sites sum to 1.0 each (roughly)
        if a_site_sum > 0 and abs(a_site_sum - 1.0) > 0.05:
            warnings.append(f"A-site fractions sum to {a_site_sum:.3f} (expected ~1.0)")
            
        if b_site_sum > 0 and abs(b_site_sum - 1.0) > 0.05:
            warnings.append(f"B-site fractions sum to {b_site_sum:.3f} (expected ~1.0)")
            
        if 'Pb' in parsed_dict:
            warnings.append("Lead (Pb) detected — this platform focuses on lead-free piezoceramics.")
            
        return warnings
