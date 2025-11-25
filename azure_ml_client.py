from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

def get_ml_client():
    load_dotenv()
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("AML_WORKSPACE_NAME")

    return MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace
    )
