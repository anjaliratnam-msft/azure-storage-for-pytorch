import os
import sys
import re

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

    pattern = r'(CONTAINER_URL\s*=\s*\(\s*"https://)([^.]+)(\.blob\.core\.windows\.net/)([^"]+)("\s*\))'
    new_url = rf"\1{STORAGE_ACCOUNT_NAME}\3{CONTAINER_NAME}\5"

    return re.sub(pattern, new_url, content)


def run_sample(path):
    modified_content = modify_url(path)
    if modified_content:
        local_vars = {}
        exec(modified_content, locals=local_vars)

        for var_name, var_value in local_vars.items():
            if var_name == "map_dataset" or var_name == "iterable_dataset":
                for item in var_value:
                    print(item)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        run_sample(path)
    else:
        print("Please provide the path to the sample file")
