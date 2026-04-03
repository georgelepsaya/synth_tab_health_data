from azure.ai.ml.entities import AmlCompute
from azure_ml_client import get_ml_client

ml_client = get_ml_client()


def ensure_compute(name, size, tier="dedicated"):
    try:
        ml_client.compute.get(name)
        print(f"Compute '{name}' already exists.")
    except Exception:
        print(f"Creating compute '{name}' ({size}, {tier})...")
        compute = AmlCompute(
            name=name,
            size=size,
            min_instances=0,
            max_instances=2,
            tier=tier,
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute '{name}' created.")


ensure_compute("cpu-cluster", "Standard_D13_v2")
ensure_compute("gpu-cluster", "Standard_NC12s_v3", tier="low_priority")
