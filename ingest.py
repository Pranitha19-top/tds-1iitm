from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    # Load the text file
    loader = TextLoader("tds_scraped_content.txt", encoding="utf-8")
    documents = loader.load()

    # Initialize HuggingFace Embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store from documents and embeddings
    db = FAISS.from_documents(documents, embeddings)

    # Save the index locally
    db.save_local("faiss_index")

    print("✅ Ingestion complete. FAISS index saved locally as 'faiss_index'.")

if _name_ == "_main_":
    main()
