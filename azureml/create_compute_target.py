from azure.ai.ml.entities import AmlCompute
from azure_ml_client import get_ml_client

ml_client = get_ml_client()

cpu_compute_target = "cpu-cluster"

try:
    ml_client.compute.get(cpu_compute_target)
    print("Compute already exists.")
except Exception:
    print("Creating a new CPU compute target...")
    compute = AmlCompute(
        name=cpu_compute_target,
        size="Standard_D13_v2",
        min_instances=0,
        max_instances=2,
    )
    ml_client.compute.begin_create_or_update(compute).result()

print("CPU compute ready.")
