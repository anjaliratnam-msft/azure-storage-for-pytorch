import os
import random
import string

import pytest
from azstoragetorch.io import BlobIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

