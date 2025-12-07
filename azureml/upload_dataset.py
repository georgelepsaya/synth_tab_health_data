from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azureml.azure_ml_client import get_ml_client

ml_client = get_ml_client()

version = "1"

data_asset = Data(
    name="heart_disease",
    version=version,
    description="Heart Disease Dataset from UCI ML Repository",
    path="data/heart_disease.csv",
    type=AssetTypes.URI_FILE,
)

try:
    data_asset = ml_client.data.get(name="heart_disease", version=version)
    print(f"Data asset already exists. Name: {data_asset.name}, version: {data_asset.version}")
except:
    ml_client.data.create_or_update(data_asset)
    print(f"Data asset created. Name: {data_asset.name}, version: {data_asset.version}")
