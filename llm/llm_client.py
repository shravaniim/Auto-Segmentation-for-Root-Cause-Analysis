import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

API_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

client = AzureOpenAI(
    azure_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)