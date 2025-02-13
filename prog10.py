# Import libraries
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI  # Replace with Cohere or another LLM


# Step 1: Load and preprocess the IPC PDF
def load_ipc_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Step 2: Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=1000, chunk_overlap=200
)
text = load_ipc_pdf("Indian_Penal_Code.pdf")  # Replace with your PDF path
chunks = text_splitter.split_text(text)

# Step 3: Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)

# Step 4: Initialize conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = OpenAI(temperature=0)  # Replace with your LLM API key
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=vectorstore.as_retriever(), memory=memory
)

# Step 5: Chat interface
print("IPC Chatbot: Ask questions about the Indian Penal Code (type 'exit' to quit).")
while True:
    query = input("\nYou: ").strip()
    if query.lower() == "exit":
        break
    if query:
        response = qa_chain({"question": query})
        print(f"\nBot: {response['answer']}")
    else:
        print("Please enter a valid query.")

print("\nExiting...")
