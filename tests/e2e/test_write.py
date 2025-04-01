# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string
import pytest

from dataclasses import dataclass
from azstoragetorch.io import BlobIO
from azstoragetorch.exceptions import FatalBlobIOWriteError

_STAGE_BLOCK_SIZE = 32 * 1024 * 1024


@dataclass
class Blob:
    data: bytes
    url: str


@pytest.fixture(scope="module")
def small_blob(account_url, container_client):
    return write_blob_url(account_url, container_client, sample_data(20))


@pytest.fixture(scope="module")
def large_blob(account_url, container_client):
    return write_blob_url(
        account_url, container_client, sample_data(_STAGE_BLOCK_SIZE * 2)
    )


@pytest.fixture
def blob(request):
    return request.getfixturevalue(f"{request.param}_blob")


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


def write_blob_url(account_url, container_client, data):
    blob_name = random_resource_name()
    url = f"{account_url}/{container_client.container_name}/{blob_name}"

    return Blob(data=data, url=url)


def sample_data(data_length=20):
    return os.urandom(data_length)


class TestWrite:
    @pytest.mark.parametrize(
        "blob",
        [
            "small",
            "large",
        ],
        indirect=True,
    )
    def test_write_all_content(self, blob):
        with BlobIO(blob.url, "wb") as f:
            assert f.write(blob.data) == len(blob.data)
            assert f.tell() == len(blob.data)

    @pytest.mark.parametrize(
        "blob, n",
        [
            ("small", 1),
            ("small", 5),
            ("small", 20),
            ("small", 21),
            ("large", _STAGE_BLOCK_SIZE * 2),
            ("large", _STAGE_BLOCK_SIZE * 3),
        ],
        indirect=["blob"],
    )
    def test_write_content_split_up(self, blob, n):
        with BlobIO(blob.url, "wb") as f:
            for i in range(0, len(blob.data), n):
                assert f.write(blob.data[i : i + n]) == min(n, len(blob.data))
            assert f.tell() == len(blob.data)

    @pytest.mark.parametrize(
        "blob",
        [
            "small",
            "large",
        ],
        indirect=True,
    )
    def test_write_error(self, blob):
        with pytest.raises(FatalBlobIOWriteError):
            with BlobIO(blob.url, "wb", credential=False) as f:
                f.write(blob.data)
                f.flush()
            assert not f.closed
