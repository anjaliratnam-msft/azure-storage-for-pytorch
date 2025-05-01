import torch
import torchvision.models   # Install separately: ``pip install torchvision``
from azstoragetorch.io import BlobIO

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

# Sample model to save and load. Replace with your own model.
model = torchvision.models.resnet18(weights="DEFAULT")

# Save trained model to Azure Blob Storage. This saves the model weights
# to a blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "wb") as f:
    torch.save(model.state_dict(), f)

# Load trained model from Azure Blob Storage.  This loads the model weights
# from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
    model.load_state_dict(torch.load(f))