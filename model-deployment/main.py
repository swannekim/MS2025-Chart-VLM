# https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ml/azure-ai-ml
# https://github.com/Azure/azureml-examples/blob/main/sdk/python/foundation-models/huggingface/inference/text-generation-streaming/text-generation-streaming-online-endpoint.ipynb
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, OnlineRequestSettings
from azure.identity import DefaultAzureCredential

import time, sys
from dotenv import load_dotenv
import os

# .env
load_dotenv()

# get environment variables
sub_id = os.getenv("SUBSCRIPTION_ID")
rg = os.getenv("RESOURCE_GROUP")
ws_name = os.getenv("WORKSPACE_NAME")
model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION")

# follow steps here to connect to workspace: https://aka.ms/AzureML-WorkspaceHandlePython
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=sub_id,
    resource_group_name=rg,
    workspace_name=ws_name,
)
print("✅ Successfully created ML client!")
# Creating MLClient will not connect to the workspace.
# The client initialization is lazy, it will wait for the first time it needs to make a call

# Verify that the handle works correctly.
# If you ge an error here, modify your SUBSCRIPTION_ID, RESOURCE_GROUP, and WORKSPACE_NAME in the previous cell.
ws = ml_client.workspaces.get(ws_name)
print(f"✅ Workspace {ws.name} located in {ws.location} (RG: {ws.resource_group})")

# fetch model
registry_ml_client = MLClient(
    credential=DefaultAzureCredential(), registry_name="HuggingFace"
)
foundation_model = registry_ml_client.models.get(
    model_name, version=model_version
)
print(f"✅ Fetched model: {foundation_model.name}, version: {foundation_model.version}, id: {foundation_model.id}")

# Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
endpoint_name = "hf-cg-ep-" + str(
    int(time.time())
)  # endpoint name must be unique per Azure region, hence appending timestamp
endpoint = ManagedOnlineEndpoint(name=endpoint_name, description="Hugging Face ChartGemma Online Endpoint")
ml_client.begin_create_or_update(endpoint).wait()
print(f"✅ Endpoint created: {endpoint_name}")

# create deployment
# current problem:
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints?view=azureml-api-2&tabs=cli#error-resourcenotready
demo_deployment = ManagedOnlineDeployment(
    name="chartgemma-demo",
    endpoint_name=endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_NC40ads_H100_v5",  # GPU instance type
    instance_count=1,
    # environment_variables={
    #     "DTYPE": "float16",
    #     "DISABLE_CUSTOM_KERNELS": "1"
    # }  # T4 GPU는 flash attention에서 FP32를 지원하지 않으므로, 모델을 FP16으로 강제 로드해야 warmup 단계에서 dtype 오류(RuntimeError) 방지
)
ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
print("✅ Deployment created: chartgemma-demo")

# route traffic
# https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ml/azure-ai-ml/azure/ai/ml/entities/_endpoint/endpoint.py
# online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for demo deployment
endpoint.traffic = {"chartgemma-demo": 100}
ml_client.begin_create_or_update(endpoint).result()
print(f"✅ Endpoint traffic routed to deployment: chartgemma-demo")
# go to Endpoints hub in AzureML Studio to get the endpoint URL and test your endpoint
# byok bring your own key hyok hold your own key