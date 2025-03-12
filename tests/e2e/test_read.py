# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
from dataclasses import dataclass
import os
import random
import string

import pytest
from azstoragetorch.io import BlobIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


_PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024


@dataclass
class Blob:
    data: bytes
    url: str


@pytest.fixture(scope="module")
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="module")
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()


@pytest.fixture(scope="module")
def small_uploaded_blob(account_url, container_client):
    blob_name = random_resource_name()
    data = sample_data(20)
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    url = f"{account_url}/{container_client.container_name}/{blob_name}"

    return Blob(data=data, url=url)


@pytest.fixture(scope="module")
def large_uploaded_blob(account_url, container_client):
    blob_name = random_resource_name()
    data = sample_data(_PARTITIONED_DOWNLOAD_THRESHOLD * 2)
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    url = f"{account_url}/{container_client.container_name}/{blob_name}"

    return Blob(data=data, url=url)


@pytest.fixture(scope="module")
def small_with_newlines_uploaded_blob(account_url, container_client):
    blob_name = random_resource_name()
    data = sample_data(20, True)
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    url = f"{account_url}/{container_client.container_name}/{blob_name}"

    return Blob(data=data, url=url)


@pytest.fixture
def uploaded_blob(request):
    if request.param == "small_blob":
        return request.getfixturevalue("small_uploaded_blob")
    elif request.param == "large_blob":
        return request.getfixturevalue("large_uploaded_blob")
    elif request.param == "small_with_newlines_blob":
        return request.getfixturevalue("small_with_newlines_uploaded_blob")


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


def sample_data(data_length=20, newline=False):
    if newline:
        return "".join(
            random.choices(string.ascii_letters + string.digits + "\n", k=data_length)
        ).encode()
    return os.urandom(data_length)


class TestRead:
    @pytest.mark.parametrize(
        "uploaded_blob",
        [
            "small_blob",
            "large_blob",
        ],
        indirect=True,
    )
    def test_reads_all_data(self, uploaded_blob):
        blob = uploaded_blob
        with BlobIO(blob.url, "rb") as f:
            assert f.read() == blob.data

    @pytest.mark.parametrize("n", [1, 2, 4, 5, 9])
    @pytest.mark.parametrize("uploaded_blob", ["small_blob"], indirect=True)
    def test_read_n_bytes(self, uploaded_blob, n):
        blob = uploaded_blob
        with BlobIO(blob.url, "rb") as f:
            for i in range(0, 10, n):
                assert f.read(n) == blob.data[i : i + n]

    @pytest.mark.parametrize("n", [1, 2, 4, 5, 9])
    @pytest.mark.parametrize("uploaded_blob", ["small_blob"], indirect=True)
    def test_random_seeks_and_reads(self, uploaded_blob, n):
        blob = uploaded_blob
        with BlobIO(blob.url, "rb") as f:
            f.seek(n)
            assert f.read() == blob.data[n:]

    @pytest.mark.parametrize(
        "uploaded_blob", ["small_with_newlines_blob"], indirect=True
    )
    def test_read_using_iter(self, uploaded_blob):
        blob = uploaded_blob
        with BlobIO(blob.url, "rb") as f:
            data = b""
            for i in f:
                data += i
            assert data == blob.data
