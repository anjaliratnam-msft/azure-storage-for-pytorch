# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Load a PyTorch model"""

import torch
import torchvision.models  # Install separately: ``pip install torchvision``
from azstoragetorch.io import BlobIO

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# Model to load weights for. Replace with your own model.
model = torchvision.models.resnet18()

# Load trained model from Azure Blob Storage.  This loads the model weights
# from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
    model.load_state_dict(torch.load(f))
