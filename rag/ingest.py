import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load env variables including OpenAI Key
load_dotenv()

# Add parent dir to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.chroma_store import ChromaStore

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def ingest_sec_filings():
    """Reads all text files in data/ and ingests them into ChromaDB."""
    store = ChromaStore()
    
    # 500-token chunker (approximate with chars or use TokenSplitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # Approx 500 tokens (1 token ~4 chars)
        chunk_overlap=200
    )
    
    all_chunks = []
    all_metadatas = []
    
    if not os.path.exists(DB_DIR):
        print(f"Data directory {DB_DIR} does not exist.")
        return
        
    for filename in os.listdir(DB_DIR):
        if filename.endswith(".txt") and not ("sample_tickers" in filename): # Ignore static json
            filepath = os.path.join(DB_DIR, filename)
            ticker = filename.split('_')[0].upper() # naming convention e.g. NVDA_sec.txt
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                chunks = text_splitter.split_text(content)
                metadata = {
                    'ticker': ticker,
                    'source': filename,
                    'date': '2025-Q1' # placeholder or from filename
                }
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    
                print(f"[{ticker}] Chunked {filename} into {len(chunks)} segments.")
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if all_chunks:
        print(f"Adding {len(all_chunks)} total chunks into ChromaDB...")
        store.add_documents(all_chunks, all_metadatas)
        print("Ingestion complete.")
    else:
        print("No .txt files found to ingest in data/")

if __name__ == "__main__":
    ingest_sec_filings()
