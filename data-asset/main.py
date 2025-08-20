from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

from dotenv import load_dotenv
import os

# .env
load_dotenv()

# get environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace = os.getenv("WORKSPACE_NAME")
data_path = os.getenv("DATA_PATH")

# Connect to the AzureML workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
print("✅ Successfully created ML client!")
# Verify that the handle works correctly.
# If you ge an error here, modify your SUBSCRIPTION_ID, RESOURCE_GROUP, and WORKSPACE_NAME in the previous cell.
ws = ml_client.workspaces.get(workspace)
print(f"✅ Workspace {ws.name} located in {ws.location} (RG: {ws.resource_group})")

# Set the version number of the data asset (for example: '1')
# VERSION = "<VERSION>"

# Set the path, supported paths include:
# local: './<path>/<folder>' (this will be automatically uploaded to cloud storage)
# blob:  'wasbs://<container_name>@<account_name>.blob.core.windows.net/<path>/<folder>'
# ADLS gen2: 'abfss://<file_system>@<account_name>.dfs.core.windows.net/<path>/<folder>'
# Datastore: 'azureml://datastores/<data_store_name>/paths/<path>/<folder>'
data_path = os.getenv("DATA_PATH")
data_version = os.getenv("DATA_VERSION")

# Define the Data asset object
my_data = Data(
    path=data_path,
    type=AssetTypes.URI_FOLDER,
    description="https://github.com/vis-nlp/ChartQA",
    name="ChartQA-Dataset"
    #version=VERSION,
)
print("✅ Successfully created Data asset object!")

# Create the data asset in the workspace
ml_client.data.create_or_update(my_data)
print(f"✅ Successfully created {my_data.name} in Azure ML workspace!")

# Your file exceeds 100 MB. If you experience low speeds, latency, or broken connections, we recommend using the AzCopyv10 tool for this file transfer.       
# Example: azcopy copy 'C:\Users\t-yooyeunkim\OneDrive - Microsoft\Desktop\Projects\MS2025-Chart-VLLM\data-asset\ChartQA-Dataset' 'https://stsagekimaih912057810561.blob.core.windows.net/e50dcc49-4202-489d-ae4f-22f8078e2c70-azureml-blobstore/LocalUpload/e842e34bb09e3a842dfe9e69246792751cdf9522462626cbf40335d67174ea4e/ChartQA-Dataset'
# See https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10 for more information.