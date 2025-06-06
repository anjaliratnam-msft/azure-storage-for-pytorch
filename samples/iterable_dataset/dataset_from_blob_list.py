# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Creates an iterable-style dataset from a list of blobs"""

from azstoragetorch.datasets import IterableBlobDataset

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

# Create an iterable-style dataset from the list of blob URLs
iterable_dataset = IterableBlobDataset.from_blob_urls(blob_urls)
