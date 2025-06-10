# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""For development purposes only. Serves as a way to automate running of samples."""

import os
import sys
import re
import subprocess
import tempfile
import argparse

STORAGE_ACCOUNT_NAME = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = os.environ.get("AZSTORAGETORCH_CONTAINER_NAME")


def modify_url(path):
    if not STORAGE_ACCOUNT_NAME or not CONTAINER_NAME:
        print(
            "Please set environment variables AZSTORAGETORCH_STORAGE_ACCOUNT_NAME and AZSTORAGETORCH_CONTAINER_NAME"
        )
        return

    with open(path, "r") as f:
        content = f.read()

    account_pattern = r"<my-storage-account-name>"
    container_pattern = r"<my-container-name>"
    modified_content = re.sub(account_pattern, STORAGE_ACCOUNT_NAME, content)
    return re.sub(container_pattern, CONTAINER_NAME, modified_content)


def run_sample(path):
    modified_content = modify_url(path)
    if modified_content:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(modified_content)
            temp_path = temp_file.name
        try:
            result = subprocess.run(
                [sys.executable, temp_path], capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running sample: {e.stderr}")
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sample script with a modified Azure Storage URL"
    )
    parser.add_argument("path", help="Path to the sample script")
    args = parser.parse_args()
    run_sample(args.path)
