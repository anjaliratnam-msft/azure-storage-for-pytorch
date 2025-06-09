# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Creates a map-style dataset from a container url"""

from azstoragetorch.datasets import BlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# Create a map-style dataset by listing blobs in the container specified by CONTAINER_URL.
map_dataset = BlobDataset.from_container_url(CONTAINER_URL)
