from collections import OrderedDict
from typing import Dict, Any, Optional

class FeatureCache:
    """
    Simple LRU Cache for expensive atomic feature derivations.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
    def get(self, formula: str) -> Optional[Dict[str, Any]]:
        if formula not in self.cache:
            return None
            
        # Move to end to mark as recently used
        self.cache.move_to_end(formula)
        return self.cache[formula]
        
    def put(self, formula: str, features: Dict[str, Any]) -> None:
        self.cache[formula] = features
        self.cache.move_to_end(formula)
        
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
