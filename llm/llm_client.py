import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

API_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

# Must match an actual deployment name in the Azure OpenAI resource (Azure
# OpenAI Studio > Deployments) -- not necessarily the underlying model name.
# Defaults to "gpt-4.1" (the value previously hardcoded at both call sites).
MODEL_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

client = AzureOpenAI(
    azure_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)