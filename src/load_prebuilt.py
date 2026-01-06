# src/load_prebuilt.py
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
from config import PREBUILT_PARQUET, VECTOR_STORE_DIR


def load_parquet_to_chroma(batch_size=10000):
    if not PREBUILT_PARQUET.exists():
        print(f"Pre-built parquet file not found at {PREBUILT_PARQUET}")
        print("Falling back to sample store from Task 2.")
        return False

    print(f"Loading pre-built embeddings from {PREBUILT_PARQUET} in batches of {batch_size}...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db_path = VECTOR_STORE_DIR / "full_prebuilt"
    db_path.mkdir(parents=True, exist_ok=True)

    # Initialize empty DB (will add in batches)
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name="complaint_chunks"
    )

    total_chunks = 0
    try:
        # Read parquet in batches using pyarrow
        for i, batch_df in enumerate(pd.read_parquet(PREBUILT_PARQUET, engine='pyarrow').iter_batches(batch_size=batch_size)):
            print(f"Processing batch {i+1} with {len(batch_df)} chunks...")

            documents = []
            for _, row in batch_df.iterrows():
                metadata = row['metadata'] if 'metadata' in row else {}
                doc = Document(
                    page_content=row['text'],
                    metadata=metadata
                )
                documents.append(doc)

            # Add batch to Chroma
            db.add_documents(documents)
            total_chunks += len(documents)

            print(f"Indexed {total_chunks:,} chunks so far")

        print(f"\nFull pre-built vector store successfully created!")
        print(f"Total chunks indexed: {total_chunks:,}")
        print(f"Location: {db_path}")
        return True

    except Exception as e:
        print(f"Error during loading: {e}")
        return False

if __name__ == "__main__":
    success = load_parquet_to_chroma(batch_size=10000)
    if not success:
        print("\nYou can proceed with Task 3 using the sample store at vector_store/sample_chroma")