import os
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Set DB Path
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))

# Ensure directory exists
os.makedirs(DB_DIR, exist_ok=True)

class ChromaStore:
    def __init__(self, collection_name: str = "sec_filings"):
        """Initializes ChromaDB with HuggingFace all-MiniLM-L6-v2."""
        # Setup Free Local Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize LangChain Chroma Wrapper
        self.vectorstore = Chroma(
            collection_name=collection_name, 
            embedding_function=self.embeddings,
            persist_directory=DB_DIR
        )

    def add_documents(self, texts: list, metadatas: list):
        """Adds text documents and metadatas into the store."""
        if not texts:
            return
            
        docs = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(texts, metadatas)
        ]
        
        self.vectorstore.add_documents(docs)
        print(f"Added {len(docs)} documents to Chroma store.")

    def query(self, ticker: str, n_results=5) -> str:
        """Queries documents using ticker context. Returns joined string of contexts."""
        try:
            # Query by ticker
            results = self.vectorstore.similarity_search(
                query=f"Details about {ticker} financials, risks, and performance",
                k=n_results
            )
            
            if not results:
                print(f"No Chroma results for {ticker}")
                return f"No SEC filings context available for {ticker}."
                
            contexts = [doc.page_content for doc in results]
            return "\n---\n".join(contexts)
            
        except Exception as e:
            print(f"Error querying Chroma for {ticker}: {e}")
            return f"Error retrieving context for {ticker}."

# Singleton
store = ChromaStore()

if __name__ == "__main__":
    # Test query
    print(store.query("NVDA"))
