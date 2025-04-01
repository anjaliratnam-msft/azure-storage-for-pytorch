# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import pytest
import torch

from azstoragetorch.io import BlobIO


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
def state_dict_blob_url(account_url, container_client, tmp_path_factory):
    model_name = tmp_path_factory.mktemp("model") / "model.pth"
    return f"{account_url}/{container_client.container_name}/{model_name.name}"


class TestSave:
    def test_save_model(self, model, state_dict_blob_url):
        with BlobIO(state_dict_blob_url, "wb") as f:
            torch.save(model.state_dict(), f)

        with BlobIO(state_dict_blob_url, "rb") as f:
            state_dict = torch.load(f)

        assert state_dict.keys() == model.state_dict().keys()

        for key, value in model.state_dict().items():
            assert torch.equal(state_dict[key], value)
