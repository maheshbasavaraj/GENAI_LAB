# Import libraries
from langchain.chains import LLMChain
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os


# Step 1: Authenticate Google Drive
def load_google_doc():
    # Set up Google Drive API
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # Load a text file from Google Drive
    service = build("drive", "v3", credentials=creds)
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your file ID
    result = service.files().get(fileId=file_id).execute()
    content = service.files().export(fileId=file_id, mimeType="text/plain").execute()
    return content.decode("utf-8")


# Step 2: Initialize Cohere LLM
cohere_api_key = os.getenv("COHERE_API_KEY")  # Or paste directly (not recommended)
llm = Cohere(cohere_api_key=cohere_api_key, temperature=0.5)

# Step 3: Create a Prompt Template
template = """
You are a helpful assistant. Analyze the following document and answer the query:

DOCUMENT:
{document}

QUERY:
{query}

ANSWER:
"""
prompt = PromptTemplate(input_variables=["document", "query"], template=template)

# Step 4: Run the QA Chain
document_text = load_google_doc()  # Load from Google Drive
query = "Summarize the key points of this document."

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(document=document_text, query=query)

print("\n=== Response ===")
print(response)
