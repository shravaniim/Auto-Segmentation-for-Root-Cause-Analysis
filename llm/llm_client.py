from openai import AzureOpenAI

API_ENDPOINT = "https://your-azure-openai-endpoint.openai.azure.com/"  # Replace with your Azure OpenAI endpoint
API_KEY = "AZURE_OPENAI_API_KEY"
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    azure_endpoint=API_ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)