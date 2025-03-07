# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string

import pytest
from azstoragetorch.io import BlobIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


@pytest.fixture
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture
def sample_data():
    return os.urandom(10)


@pytest.fixture
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()


@pytest.fixture
def blob_client(container_client, sample_data):
    blob_name = random_resource_name()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(sample_data)
    return blob_client


@pytest.fixture
def blob_url(account_url, container_client, blob_client):
    return f"{account_url}/{container_client.container_name}/{blob_client.blob_name}"


@pytest.fixture
def blob_io(blob_url):
    yield BlobIO(blob_url=blob_url, mode="rb")


def random_resource_name(name_length=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=name_length))


class TestRead:
    def test_reads_all_data(self, blob_io, sample_data):
        with blob_io as f:
            assert f.read() == sample_data
