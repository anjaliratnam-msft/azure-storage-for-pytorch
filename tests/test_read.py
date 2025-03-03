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


EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE = 4 * 1024 * 1024


@pytest.fixture
def sas_token():
    return "sp=r&st=2024-10-28T20:22:30Z&se=2024-10-29T04:22:30Z&spr=https&sv=2022-11-02&sr=c&sig=signature"


@pytest.fixture
def blob():
    return BlobIO(
        blob_url="https://anjaliratnamtest.blob.core.windows.net/test/test.txt",
        mode="rb"
    )

class TestRead:
    def test_reads_all_data(
        self, blob
    ):
        with blob as f:
            assert f.read() == b"Hello World"

    