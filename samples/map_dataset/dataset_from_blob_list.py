# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Creates a map-style dataset from a list of blobs"""

from azstoragetorch.datasets import BlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# List of blob URLs to create dataset from. Update with your own blob names.
blob_urls = [
    f"{CONTAINER_URL}/<blob-name-1>",
    f"{CONTAINER_URL}/<blob-name-2>",
    f"{CONTAINER_URL}/<blob-name-3>",
]

# Create a map-style dataset from the list of blob URLs
map_dataset = BlobDataset.from_blob_urls(blob_urls)
