from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure_ml_client import get_ml_client

ml_client = get_ml_client()

data_asset = Data(
    name="heart_disease_npy",
    description="Heart Disease Dataset split into NPYs",
    path="data/heart_disease_npy",
    type=AssetTypes.URI_FOLDER,
)

ml_client.data.create_or_update(data_asset)
print(f"Created asset {data_asset.name}:{data_asset.version}")
