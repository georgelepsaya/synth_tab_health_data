from azure.ai.ml.entities import Environment
from azure_ml_client import get_ml_client

ml_client = get_ml_client()

env_name = "tabddpm"

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04",
    conda_file="../environments/tabddpm_env.yml",
    name=env_name,
    description="TabDDPM conda environment for synthetic data generation",
)
ml_client.environments.create_or_update(env_docker_conda)

envs = ml_client.environments.list()

print(f"Environment {env_name} created")
