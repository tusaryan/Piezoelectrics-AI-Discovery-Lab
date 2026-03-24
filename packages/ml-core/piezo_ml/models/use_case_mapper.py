import math
from typing import Dict, Any, List

class UseCaseMapper:
    """
    Maps fundamental piezo properties (d33, Tc, hardness) to real-world commercial applications
    using a normalized geometric distance algorithm to ideal domain centroids instead of restrictive hardcoding.
    """
    
    # Define the ideal scientific "centroids" for each operational domain.
    # d33 in pC/N, Tc in Celsius, Vickers Hardness in HV.
    # Normalization weights allow adjusting sensitivity per property.
    DOMAINS: List[Dict[str, Any]] = [
        {
            "use_case": "High-Temp Actuator",
            "description": "Suitable for aerospace engine monitoring and high-stress industrial environments due to massive Curie boundary.",
            "recommended_applications": ["Fuel Injection Systems", "Turbine Sensors", "Deep Well Drilling"],
            "centroid": {"d33": 150.0, "tc": 600.0, "hardness": 600.0},
            "icon": "zap",
            "weights": {"d33": 1.0, "tc": 3.0, "hardness": 1.0} # Tc is extremely critical here
        },
        {
            "use_case": "Medical Ultrasound Transducer",
            "description": "Exceptional piezoelectric coefficient enables extremely high-resolution pulse-echo acoustic rendering.",
            "recommended_applications": ["Diagnostic Ultrasound", "NDT Arrays", "Micro-actuators"],
            "centroid": {"d33": 600.0, "tc": 150.0, "hardness": 400.0},
            "icon": "activity",
            "weights": {"d33": 3.0, "tc": 1.0, "hardness": 0.5} # d33 is paramount
        },
        {
            "use_case": "Heavy-Duty Sonar Array",
            "description": "Extreme mechanical strength paired with reasonable coupling makes this ideal for high-pressure deep sea exposure.",
            "recommended_applications": ["Naval Sonar", "Hydrophone Arrays", "Ballistic Sensors"],
            "centroid": {"d33": 250.0, "tc": 200.0, "hardness": 1200.0},
            "icon": "shield",
            "weights": {"d33": 1.5, "tc": 1.0, "hardness": 3.0} # Hardness is paramount
        },
        {
            "use_case": "Energy Harvester",
            "description": "Good balance of thermal stability and energy conversion efficiency for scavenging mechanical vibrations.",
            "recommended_applications": ["IoT Wearable Power", "Bridge Vibration Scavenger", "Automotive Shocks"],
            "centroid": {"d33": 350.0, "tc": 250.0, "hardness": 500.0},
            "icon": "battery-charging",
            "weights": {"d33": 2.0, "tc": 1.5, "hardness": 1.0}
        },
        {
            "use_case": "General Purpose Ceramic",
            "description": "Modest properties suitable for low-cost, bulk commercial components without extreme stress envelopes.",
            "recommended_applications": ["Buzzer Elements", "Gas Igniters", "Basic Sensors"],
            "centroid": {"d33": 100.0, "tc": 300.0, "hardness": 500.0},
            "icon": "radio",
            "weights": {"d33": 1.0, "tc": 1.0, "hardness": 1.0}
        }
    ]
    
    # Global scaling factors to normalize raw values before geometric distance calculation
    SCALE = {"d33": 1000.0, "tc": 1000.0, "hardness": 2000.0}

    @classmethod
    def classify(cls, d33: float, tc: float, vickers_hardness: float = -1.0) -> Dict[str, Any]:
        """
        Classifies material dynamically using normalized weighted Euclidean distance
        to the predefined operational domain centroids.
        """
        best_domain = None
        min_distance = float('inf')
        
        # If hardness wasn't predicted (e.g., failed or missing), impute a safe mean 
        # so the model still evaluates purely on Piezo properties.
        active_hardness = vickers_hardness if vickers_hardness > 0 else 500.0
        
        for domain in cls.DOMAINS:
            centroid = domain["centroid"]
            weights = domain["weights"]
            
            # Calculate normalized weighted distance
            dist_d33 = weights["d33"] * ((d33 - centroid["d33"]) / cls.SCALE["d33"]) ** 2
            dist_tc = weights["tc"] * ((tc - centroid["tc"]) / cls.SCALE["tc"]) ** 2
            dist_hard = weights["hardness"] * ((active_hardness - centroid["hardness"]) / cls.SCALE["hardness"]) ** 2
            
            # Penalize heavily if the material falls severely below the required threshold for a key property
            # For example, a High-Temp Actuator MUST have High TC. If it doesn't, increase distance.
            penalty = 0.0
            if d33 < centroid["d33"] * 0.5: penalty += weights["d33"] * 2.0
            if tc < centroid["tc"] * 0.5: penalty += weights["tc"] * 2.0
            if active_hardness < centroid["hardness"] * 0.5: penalty += weights["hardness"] * 2.0
            
            total_distance = math.sqrt(dist_d33 + dist_tc + dist_hard) + penalty
            
            if total_distance < min_distance:
                min_distance = total_distance
                best_domain = domain
                
        # Calculate a pseudo-confidence score inversely proportional to distance (max 0.99)
        confidence = max(0.40, min(0.99, 1.0 - min_distance))
        
        result = best_domain.copy()
        del result["centroid"]
        del result["weights"]
        result["confidence"] = round(confidence, 2)
        
        return result
