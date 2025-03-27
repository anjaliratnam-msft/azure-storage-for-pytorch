# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string
import pytest
import torch

from azstoragetorch.io import BlobIO

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


@pytest.fixture(scope="package")
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="package")
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


@pytest.fixture(scope="module", autouse=True)
def torch_hub_cache(tmp_path_factory):
    current_dir = torch.hub.get_dir()
    torch.hub.set_dir(tmp_path_factory.mktemp("torch_hub"))
    yield
    torch.hub.set_dir(current_dir)


@pytest.fixture(scope="module")
def model():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", pretrained=False)
    return model


@pytest.fixture(scope="module")
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("model")


@pytest.fixture(scope="module")
def state_dict_blob_url(account_url, container_client, tmp_path):
    blob_name = tmp_path / "model.pth"
    return f"{account_url}/{container_client.container_name}/{blob_name.name}"


class TestSave:
    def test_save_model(self, model, state_dict_blob_url):
        with BlobIO(state_dict_blob_url, "wb") as f:
            torch.save(model.state_dict(), f)

        with BlobIO(state_dict_blob_url, "rb") as f:
            state_dict = torch.load(f)
            
        assert state_dict.keys() == model.state_dict().keys()

        for key, value in model.state_dict().items():
            assert torch.equal(state_dict[key], value)