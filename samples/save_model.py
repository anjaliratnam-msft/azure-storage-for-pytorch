# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Save a PyTorch model"""

import torch
import torchvision.models  # Install separately: ``pip install torchvision``
from azstoragetorch.io import BlobIO

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# Model to save. Replace with your own model.
model = torchvision.models.resnet18(weights="DEFAULT")

# Save trained model to Azure Blob Storage. This saves the model weights
# to a blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "wb") as f:
    torch.save(model.state_dict(), f)
