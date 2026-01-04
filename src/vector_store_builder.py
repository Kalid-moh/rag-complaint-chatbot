# src/vector_store_builder.py
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # This wrapper requires 'chromadb' to be installed
from langchain_core.documents import Document
from pathlib import Path
from .config import FILTERED_CSV, VECTOR_STORE_DIR


class SampleVectorStoreBuilder:
    def __init__(self, sample_size: int = 12000):
        self.sample_size = sample_size
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vector_store_path = VECTOR_STORE_DIR / "sample_chroma"

    def load_filtered_data(self) -> pd.DataFrame:
        print(f"Loading filtered complaints from {FILTERED_CSV}...")
        df = pd.read_csv(FILTERED_CSV)
        print(f"Loaded {len(df):,} complaints")
        return df

    def create_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\nCreating stratified sample of {self.sample_size} complaints...")
        # Stratified sampling proportional to product_category
        sample_df = df.groupby('product_category', group_keys=False).apply(
            lambda x: x.sample(frac=self.sample_size / len(df), random_state=42)
        )
        if len(sample_df) > self.sample_size:
            sample_df = sample_df.sample(n=self.sample_size, random_state=42)
        
        print(f"Sample created: {len(sample_df)} complaints")
        print("Distribution:")
        print(sample_df['product_category'].value_counts())
        return sample_df

    def chunk_narratives(self, sample_df: pd.DataFrame) -> list[Document]:
        print("\nChunking narratives (chunk_size=500, overlap=50)...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        documents = []
        for _, row in sample_df.iterrows():
            chunks = splitter.split_text(row['clean_narrative'])
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "complaint_id": str(row.get('Complaint ID', 'unknown')),
                        "product_category": row['product_category'],
                        "product": row['Product'],
                        "issue": row.get('Issue', ''),
                        "sub_issue": row.get('Sub-issue', ''),
                        "company": row.get('Company', ''),
                        "state": row.get('State', ''),
                        "date_received": row['Date received'],
                        "chunk_index": idx,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)

        print(f"Created {len(documents)} chunks from {len(sample_df)} complaints")
        return documents

    def build_and_persist_vector_store(self, documents: list[Document]):
        print("\nBuilding vector store with ChromaDB (auto-persistence enabled)...")
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # No need for db.persist() — it's automatic with persist_directory
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.vector_store_path),
            collection_name="complaint_chunks"
        )
        print(f"Vector store automatically persisted to: {self.vector_store_path}")

    def run(self):
        df = self.load_filtered_data()
        sample_df = self.create_stratified_sample(df)
        documents = self.chunk_narratives(sample_df)
        self.build_and_persist_vector_store(documents)
        print("\nTask 2 completed successfully!")


if __name__ == "__main__":
    builder = SampleVectorStoreBuilder(sample_size=12000)
    builder.run()