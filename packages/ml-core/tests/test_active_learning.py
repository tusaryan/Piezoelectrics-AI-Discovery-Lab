import pytest
import asyncio
from piezo_ml.optimization.active_learning import ActiveLearningSimulator

@pytest.mark.asyncio
async def test_active_learning_simulator_ucb():
    simulator = ActiveLearningSimulator(total_budget=10, strategy="UCB")
    
    events = []
    async def cb(msg):
        events.append(msg)
        
    result = await simulator.simulate_async(cb=cb)
    
    assert "strategy_curve" in result
    assert "baseline_curve" in result
    assert len(result["strategy_curve"]) == 10
    assert len(events) >= 1  # Should emit some progress events
    assert result["efficiency_gain"] >= 0.0
    assert result["final_max_d33"] > 100.0

@pytest.mark.asyncio
async def test_active_learning_simulator_random():
    simulator = ActiveLearningSimulator(total_budget=5, strategy="Random")
    result = await simulator.simulate_async(cb=None)
    
    assert len(result["baseline_curve"]) == 5
    assert result["iterations_to_max"]["baseline"] == 5
