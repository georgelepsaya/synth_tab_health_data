from azure.ai.ml.entities import Environment
from azure_ml_client import get_ml_client

ml_client = get_ml_client()

env_name = "synthtabhealthdata"

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
    conda_file="../environments/synthcity_env.yml",
    name=env_name,
    description="Synthcity GPU environment for synthetic data generation and evaluation",
)
ml_client.environments.create_or_update(env_docker_conda)

print(f"Environment '{env_name}' registered")
