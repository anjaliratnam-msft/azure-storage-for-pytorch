# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Creates an iterable-style dataset using a prefix"""

from azstoragetorch.datasets import IterableBlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = (
    "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
)

# Create an iterable-style dataset only including blobs whose name starts with the prefix "images/"
iterable_dataset = IterableBlobDataset.from_container_url(
    CONTAINER_URL, prefix="images/"
)
