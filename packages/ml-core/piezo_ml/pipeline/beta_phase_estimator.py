import numpy as np

class BetaPhaseEstimator:
    """
    Estimates beta-phase percentage in PVDF based on filler_wt_pct.
    Utilizes a simple polynomial regression rule of thumb derived from literature for PVDF composites.
    Generally, addition of ceramic fillers like BaTiO3 up to ~10-15 wt% increases β-phase drastically,
    then plateaus or falls due to agglomeration defects.
    """
    
    @staticmethod
    def estimate(matrix_type: str, filler_wt_pct: float) -> float:
        if not matrix_type or "pvdf" not in matrix_type.lower():
            return 0.0
            
        # Base PVDF beta phase casted without filler is roughly ~20-30% depending on method.
        # Here we model a curve based on empirical data: rises to peak around 12 wt%, then drops.
        
        wt = float(filler_wt_pct)
        if wt <= 0:
            return 25.0
            
        # Simplistic polynomial curve fitting known composite behavior:
        # y = -0.15x^2 + 3.8x + 25
        # peak is at x ~= 12.6 wt%, y ~= 49%
        
        estimated_beta = -0.15 * (wt ** 2) + 3.8 * wt + 25.0
        
        # Clamp to realistic physical bounds (0 -> 100)
        return float(np.clip(estimated_beta, 10.0, 95.0))
