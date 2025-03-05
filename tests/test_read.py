# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string
from unittest import mock
import pytest

from azure.core.credentials import AzureSasCredential, AzureNamedKeyCredential
from azure.identity import DefaultAzureCredential

from azstoragetorch.io import BlobIO
from azstoragetorch._client import AzStorageTorchBlobClient

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient



@pytest.fixture
def account_url():
    account_name = os.environ.get('PYTORCH_STORAGE_ACCOUNT_NAME', None) or "pytorchstorageaccount"
    return "https://" + account_name.strip('"') + ".blob.core.windows.net"

@pytest.fixture
def container_name():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

@pytest.fixture
def blob_name():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

@pytest.fixture
def sample_data():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)).encode()

@pytest.fixture
def container(
    account_url, container_name
):
    blob_service_client = BlobServiceClient(account_url, credential=DefaultAzureCredential())

    cont = blob_service_client.create_container(name=container_name)
    yield blob_service_client
    cont.delete_container()

@pytest.fixture
def blob(
    account_url, container_name, blob_name, sample_data, container
):  
    blob_client = container.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(sample_data, blob_type="BlockBlob")

    yield BlobIO(
        blob_url=account_url + "/" + container_name + "/" + blob_name,
        mode="rb"
    )


class TestRead:
    def test_reads_all_data(
        self, blob, sample_data
    ):
        with blob as f:
            assert f.read() == sample_data

    