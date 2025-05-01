from azstoragetorch.datasets import BlobDataset


def create_dataset_from_container_url(CONTAINER_URL, prefix=None):
    if prefix is None:
        map_dataset = BlobDataset.from_container_url(CONTAINER_URL)
    else:
        map_dataset = BlobDataset.from_container_url(CONTAINER_URL, prefix=prefix)

def main(account_name, container_name):
    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Create a map-style dataset by listing blobs in the container specified by CONTAINER_URL.
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL)

    # Create a map-style dataset only including blobs whose name starts with the prefix "images/"
    map_dataset = BlobDataset.from_container_url(CONTAINER_URL, prefix="images/")

    # List of blob URLs to create dataset from. Update with your own blob names.
    blob_urls = [
        f"{CONTAINER_URL}/<blob-name-1>",
        f"{CONTAINER_URL}/<blob-name-2>",
        f"{CONTAINER_URL}/<blob-name-3>",
    ]

    # Create a map-style dataset from the list of blob URLs
    map_dataset = BlobDataset.from_blob_urls(blob_urls)


if __name__ == "__main__":
    account_name = "<my-storage-account-name>"
    container_name = "<my-container-name>"
    main(account_name, container_name)