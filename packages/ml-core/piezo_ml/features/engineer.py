from typing import Dict, List, Tuple
import numpy as np
from mendeleev import element
from piezo_ml.parsers.formula_parser import FormulaParser
from piezo_ml.features.feature_cache import FeatureCache

class UnsupportedElementError(Exception):
    def __init__(self, elements: List[str]):
        super().__init__(f"Unsupported elements detected: {', '.join(elements)}")
        self.elements = elements

class FeatureEngineer:
    """
    Transforms parsed chemical formulas into 35-dimensional feature vectors.
    """
    FEATURE_VERSION = "v2"
    
    # 25 most common elements in our dataset scope
    ELEMENTS = [
        'Ag', 'Ba', 'Bi', 'Ca', 'Cu', 'Eu', 'Fe', 'Gd', 'Hf', 'K', 
        'La', 'Li', 'Mg', 'Mn', 'Na', 'Nb', 'Nd', 'Pr', 'Sb', 'Sm', 
        'Sn', 'Sr', 'Ta', 'Ti', 'Zr'
    ]
    
    # 12 physics descriptors
    DESCRIPTORS = [
        'mean_atomic_mass', 'var_atomic_mass',
        'mean_electronegativity', 'var_electronegativity',
        'mean_atomic_radius', 'var_atomic_radius',
        'mean_melting_point', 'var_melting_point',
        'mean_electron_affinity', 'tolerance_factor',
        'mean_valence', 'var_valence'
    ]
    
    def __init__(self):
        self._cache = FeatureCache(capacity=10000)
        self._element_props = {}
    
    def _get_atomic_props(self, symbol: str) -> Dict[str, float]:
        if symbol not in self._element_props:
            el = element(symbol)
            self._element_props[symbol] = {
                'mass': el.atomic_weight or 0.0,
                'en': el.en_pauling or 0.0,
                'radius': el.atomic_radius or 0.0,
                'melt': el.melting_point or 0.0,
                'ea': el.electron_affinity or 0.0,
                'val': getattr(el, 'group_id', 0) or 0 # Proxy for valence
            }
        return self._element_props[symbol]

    def compute_features(self, formula: str) -> Tuple[np.ndarray, List[str]]:
        """
        Takes formula string -> Returns 35d numpy vector and feature names
        """
        cached = self._cache.get(formula)
        if cached:
            return np.array(cached['vector']), cached['names']
            
        parsed = FormulaParser.parse(formula)
        
        # Guardrail check for unsupported elements
        allowed = set(self.ELEMENTS + ['O'])
        found = set(parsed.keys())
        unsupported = list(found - allowed)
        if unsupported:
            raise UnsupportedElementError(unsupported)
            
        total_atoms = sum(v for k, v in parsed.items() if k != 'O') # Excluding O for descriptors often
        
        # 1. Elemental fractions (25d)
        vector = []
        for el in self.ELEMENTS:
            vector.append(parsed.get(el, 0.0) / total_atoms if total_atoms > 0 else 0)
            
        # 2. Physics descriptors (10d)
        props_list = []
        weights = []
        
        for symbol, count in parsed.items():
            if symbol == 'O': 
                continue
            props_list.append(self._get_atomic_props(symbol))
            weights.append(count)
            
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        masses = np.array([p['mass'] for p in props_list])
        ens = np.array([p['en'] for p in props_list])
        radii = np.array([p['radius'] for p in props_list])
        melts = np.array([p['melt'] for p in props_list])
        eas = np.array([p['ea'] for p in props_list])
        vals = np.array([p['val'] for p in props_list])
        
        # Compute means and variances
        vector.extend([
            np.average(masses, weights=weights) if len(masses) else 0,
            np.average((masses - np.average(masses, weights=weights))**2, weights=weights) if len(masses) else 0,
            np.average(ens, weights=weights) if len(ens) else 0,
            np.average((ens - np.average(ens, weights=weights))**2, weights=weights) if len(ens) else 0,
            np.average(radii, weights=weights) if len(radii) else 0,
            np.average((radii - np.average(radii, weights=weights))**2, weights=weights) if len(radii) else 0,
            np.average(melts, weights=weights) if len(melts) else 0,
            np.average((melts - np.average(melts, weights=weights))**2, weights=weights) if len(melts) else 0,
            np.average(eas, weights=weights) if len(eas) else 0,
            np.average(vals, weights=weights) if len(vals) else 0,
            np.average((vals - np.average(vals, weights=weights))**2, weights=weights) if len(vals) else 0,
        ])
        
        # Simple Goldschmidt tolerance factor approximation
        # t = (rA + rO) / (sqrt(2) * (rB + rO))
        # This requires separating A and B sites - we use an approximation based on radius size
        if len(radii) >= 2:
            rA = np.max(radii)
            rB = np.min(radii)
            rO = element('O').atomic_radius or 60.0 # ~60pm for oxygen
            tolerance = (rA + rO) / (np.sqrt(2) * (rB + rO))
        else:
            tolerance = 0.0
            
        vector.append(tolerance)
        
        feature_names = self.ELEMENTS + self.DESCRIPTORS
        
        # Cache and return
        self._cache.put(formula, {
            'vector': vector,
            'names': feature_names
        })
        
        return np.array(vector), feature_names
