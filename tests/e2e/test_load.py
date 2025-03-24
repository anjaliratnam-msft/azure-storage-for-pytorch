# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import pytest
import torch
import ssl
import functools
import certifi

from azstoragetorch.io import BlobIO


@pytest.fixture(scope="module")
def ssl_inject_certifi(monkeypatch):
    # Sets the global variable for the SSL certificates file path since torch.hub doesn't use the certifi package
    monkeypatch.setattr(
        ssl,
        "_create_default_https_context",
        functools.partial(ssl._create_default_https_context, cafile=certifi.where()),
    )


@pytest.fixture(scope="module")
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("torch_hub")


@pytest.fixture(scope="module")
def model(tmp_path):
    torch.hub.set_dir(tmp_path)
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet101", pretrained=False)
    return model


@pytest.fixture(scope="module")
def upload_model(model, container_client):
    blob_name = "model.pth"
    torch.save(model.state_dict(), blob_name)
    blob_client = container_client.get_blob_client(blob=blob_name)
    with open(blob_name, "rb") as f:
        blob_client.upload_blob(f)
    return blob_name


@pytest.fixture()
def blob_url(account_url, container_client, upload_model):
    return f"{account_url}/{container_client.container_name}/{upload_model}"


class TestLoad:
    def test_load_existing_model(self, blob_url, model):
        with BlobIO(blob_url, "rb") as f:
            state_dict = torch.load(f)

        assert state_dict.keys() == model.state_dict().keys()

        for key, value in model.state_dict().items():
            assert torch.equal(state_dict[key], value)
