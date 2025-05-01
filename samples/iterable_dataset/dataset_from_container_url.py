# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Creates an iterable-style dataset from a container url"""

from azstoragetorch.datasets import IterableBlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# Create an iterable-style dataset by listing blobs in the container specified by CONTAINER_URL.
iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)
