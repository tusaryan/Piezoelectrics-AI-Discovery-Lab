"""
Direct Celery task dispatch test.
Run from project root with venv activated:
  source apps/api/.venv/bin/activate && python test_celery.py
"""
import os, sys, time
from dotenv import load_dotenv
load_dotenv(".env")

from apps.api.app.core.celery_app import celery_app

print("=" * 60)
print("CELERY DIAGNOSTIC TEST")
print("=" * 60)
print(f"Broker URL:  {celery_app.conf.broker_url}")
print(f"Backend URL: {celery_app.conf.result_backend}")

# Verify Redis is reachable
try:
    import redis
    r = redis.from_url(celery_app.conf.broker_url)
    r.ping()
    print("Redis PING:  OK ✓")
except Exception as e:
    print(f"Redis PING:  FAILED ✗ -> {e}")
    sys.exit(1)

# Check registered tasks
print(f"\nLocal registered tasks:")
for t in sorted(celery_app.tasks.keys()):
    print(f"  - {t}")

# Check remote workers
i = celery_app.control.inspect()
print(f"\nActive workers:    {i.active()}")
print(f"Registered tasks:  {i.registered()}")

# Try dispatching a real task (small dataset that will fail quickly but prove dispatch works)
print("\n--- Sending a test task ---")
result = celery_app.send_task(
    "piezo_ai.train_model",
    kwargs={
        "job_id": "test-debug-000",
        "dataset_id": "nonexistent-dataset",
        "target": "tc",
        "model_name": "XGBoost",
        "mode": "expert",
        "params": {},
        "use_optuna": False,
        "optuna_trials": 0
    }
)
print(f"AsyncResult ID: {result.id}")
print(f"Initial state:  {result.state}")

# Wait up to 10 seconds for the worker to pick it up
print("Waiting for worker to process (10s timeout)...")
for i in range(20):
    time.sleep(0.5)
    state = result.state
    print(f"  [{i*0.5:.1f}s] state={state}")
    if state in ("SUCCESS", "FAILURE"):
        break

print(f"\nFinal state: {result.state}")
if result.state == "FAILURE":
    print(f"Error: {result.result}")
elif result.state == "SUCCESS":
    print(f"Result: {result.result}")
elif result.state == "PENDING":
    print("⚠️  Task is still PENDING after 10s — worker may not be picking up tasks!")
    print("   Check: are you running the worker from the project ROOT?")
    print("   Command: celery -A apps.api.app.core.celery_app worker --loglevel=info")
    
    # Check queue length in Redis
    queue_len = r.llen("celery")
    print(f"   Redis 'celery' queue length: {queue_len}")
