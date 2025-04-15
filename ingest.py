import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

# Define persistence directory
persist_directory = "db"

# Define updated Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",  # Ensure DuckDB + Parquet is used
    persist_directory=persist_directory,
    anonymized_telemetry=False
)

def main():
    documents = []
    
    # Load PDF documents
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize ChromaDB persistent client with correct settings
    client = chromadb.PersistentClient(settings=CHROMA_SETTINGS)

    # Create or get a collection
    collection = client.get_or_create_collection(name="pdf_documents")

    # Get existing IDs to prevent duplicates
    existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()

    # Add documents to ChromaDB collectionz
    for i, text in enumerate(texts):
        if str(i) not in existing_ids:  # Avoid duplicate inserts
            collection.add(
                ids=[str(i)],
                documents=[text.page_content],  # Extract text content
                metadatas=[text.metadata]
            )

    print("✅ PDF documents successfully processed and stored in ChromaDB!")

if __name__ == "__main__":
    main()
