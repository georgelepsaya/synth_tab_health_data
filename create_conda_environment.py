from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

load_dotenv()

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace = os.getenv("AML_WORKSPACE_NAME")

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04",
    conda_file="environment.yml",
    name="synthtabhealthdata",
    description="Conda environment for synthetic data generation",
)
ml_client.environments.create_or_update(env_docker_conda)

envs = ml_client.environments.list()
for env in envs:
    print(env.name)
