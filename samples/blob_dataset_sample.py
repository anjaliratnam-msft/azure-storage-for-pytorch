# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: blob_dataset_sample.py
DESCRIPTION:
    This sample demonstrates the ways to create a map-style dataset from Azure Blob Storage using the BlobDataset class.
USAGE: python blob_dataset_sample.py
    Set the environment variables with your own values before running the sample.

"""

from azstoragetorch.datasets import BlobDataset


def create_dataset_from_container_url(my_storage_account_name, my_container_name):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"

    # Create a map-style dataset by listing blobs in the container specified by CONTAINER_URL.
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL)
    print(map_dataset[0])


def create_dataset_from_container_url_with_prefix(my_storage_account_name, my_container_name, prefix):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"

    # Create a map-style dataset only including blobs whose name starts with the prefix "images/"
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL, prefix="images/")


def create_dataset_from_blob_urls():
     # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = f"https://{my_storage_account_name}.blob.core.windows.net/{my_container_name}"
    
    # List of blob URLs to create dataset from. Update with your own blob names.
    blob_urls = [
        f"{CONTAINER_URL}/<blob-name-1>",
        f"{CONTAINER_URL}/<blob-name-2>",
        f"{CONTAINER_URL}/<blob-name-3>",
    ]

    # Create a map-style dataset from the list of blob URLs
    map_dataset = BlobDataset.from_blob_urls(blob_urls)


if __name__ == "__main__":
    account_name = "anjaliratnamtest"
    container_name = "test"
    create_dataset_from_container_url(account_name, container_name)

    # Create a map-style dataset only including blobs whose name starts with the prefix "images/"
    create_dataset_from_container_url_with_prefix(account_name, container_name, "images/")

    create_dataset_from_blob_urls()