from azstoragetorch.datasets import IterableBlobDataset

# Update URL with your own Azure Storage account and container name
CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"    

# Create an iterable-style dataset by listing blobs in the container specified by CONTAINER_URL.
iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL)

# Create an iterable-style dataset only including blobs whose name starts with the prefix "images/"
iterable_dataset = IterableBlobDataset.from_container_url(CONTAINER_URL, prefix="images/")

# List of blob URLs to create dataset from. Update with your own blob names.
blob_urls = [
    f"{CONTAINER_URL}/<blob-name-1>",
    f"{CONTAINER_URL}/<blob-name-2>",
    f"{CONTAINER_URL}/<blob-name-3>",
]

# Create an iterable-style dataset from the list of blob URLs
iterable_dataset = IterableBlobDataset.from_blob_urls(blob_urls)