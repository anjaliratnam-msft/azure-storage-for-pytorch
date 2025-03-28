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
def small_blob(account_url, container_client):
    return upload_blob(account_url, container_client, sample_data(20))


@pytest.fixture(scope="module")
def large_blob(account_url, container_client):
    return upload_blob(
        account_url, container_client, sample_data(_PARTITIONED_DOWNLOAD_THRESHOLD * 2)
    )


@pytest.fixture(scope="module")
def small_with_newlines_blob(account_url, container_client):
    return upload_blob(account_url, container_client, sample_data_with_newlines(20, 2))


@pytest.fixture
def blob(request):
    return request.getfixturevalue(f"{request.param}_blob")


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


def sample_data(data_length=20):
    return os.urandom(data_length)


def sample_data_with_newlines(data_length=20, num_lines=1):
    lines = []
    for i in range(num_lines):
        lines.append(sample_data(int(data_length / num_lines)))
    return b"\n".join(lines)


def upload_blob(account_url, container_client, data):
    blob_name = random_resource_name()
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(data)
    url = f"{account_url}/{container_client.container_name}/{blob_name}"

    return Blob(data=data, url=url)


class TestRead:
    @pytest.mark.parametrize(
        "blob",
        [
            "small",
            "large",
        ],
        indirect=True,
    )
    def test_reads_all_data(self, blob):
        with BlobIO(blob.url, "rb") as f:
            assert f.read() == blob.data
            assert f.tell() == len(blob.data)

    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_read_n_bytes(self, small_blob, n):
        with BlobIO(small_blob.url, "rb") as f:
            for i in range(0, len(small_blob.data), n):
                assert f.read(n) == small_blob.data[i : i + n]
                expected_position = min(i + n, len(small_blob.data))
                assert f.tell() == expected_position

    @pytest.mark.parametrize("n", [1, 5, 20, 21])
    def test_random_seeks_and_reads(self, small_blob, n):
        with BlobIO(small_blob.url, "rb") as f:
            f.seek(n)
            assert f.read() == small_blob.data[n:]
            expected_position = max(n, len(small_blob.data))
            assert f.tell() == expected_position

    def test_read_using_iter(self, small_with_newlines_blob):
        with BlobIO(small_with_newlines_blob.url, "rb") as f:
            lines = [line for line in f]
            expected_lines = small_with_newlines_blob.data.splitlines(keepends=True)
            assert lines == expected_lines
