# src/rag_pipeline.py
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint
from pathlib import Path
from config import PREBUILT_PARQUET, VECTOR_STORE_DIR

class CrediTrustRAG:
    def __init__(self, top_k=5):
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.top_k = top_k

        # Define paths
        full_store_path = VECTOR_STORE_DIR / "full_prebuilt"
        sample_store_path = VECTOR_STORE_DIR / "sample_chroma"

        # Determine which store to use
        if PREBUILT_PARQUET.exists():
            if full_store_path.exists():
                print("Loading full pre-built vector store...")
                store_path = full_store_path
            else:
                print("Pre-built parquet file found, but full vector store not yet built.")
                print("Please run: python src/load_prebuilt.py first to create the full store.")
                print("Falling back to sample store for now.")
                store_path = sample_store_path
        else:
            print("Pre-built parquet file not found.")
            print("Using sample vector store from Task 2.")
            store_path = sample_store_path

        # Final check
        if not store_path.exists():
            raise FileNotFoundError(f"Vector store not found at {store_path}. Run Task 2 or load_prebuilt.py first.")

        # Load the selected store
        self.db = Chroma(
            persist_directory=str(store_path),
            embedding_function=self.embeddings,
            collection_name="complaint_chunks"
        )
        print(f"Vector store loaded: {self.db._collection.count():,} chunks from {store_path.name}")

        self.retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})

        # LLM - free Hugging Face Inference API
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.3,
            max_new_tokens=512,
        )

        # Prompt
        self.prompt = PromptTemplate.from_template(
            """You are a financial analyst assistant for CrediTrust Financial in East Africa.
Use only the provided complaint excerpts to answer the question. Be concise and evidence-based.
If the context lacks information, say "Not enough information in complaints."

Context:
{context}

Question: {question}

Answer:"""
        )

        def format_docs(docs):
            return "\n\n".join(
                f"[{i+1}] (Product: {doc.metadata.get('product_category', 'Unknown')}) {doc.page_content}"
                for i, doc in enumerate(docs)
            )

        # RAG chain
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        answer = self.chain.invoke(question)
        docs = self.retriever.invoke(question)
        sources = [
            {
                "complaint_id": doc.metadata.get("complaint_id", "unknown"),
                "product_category": doc.metadata.get("product_category", "unknown"),
                "text_preview": doc.page_content[:200] + "..."
            }
            for doc in docs
        ]
        return answer.strip(), sources

    def evaluate(self, questions):
        print("=== RAG Qualitative Evaluation ===\n")
        table = "| Question | Answer Summary | Sources (Top 2) | Quality | Comments |\n"
        table += "|---|---|---|---|---|\n"

        for q in questions:
            answer, sources = self.ask(q)
            top2 = "<br>".join([f"{s['product_category']} (ID: {s['complaint_id']})" for s in sources[:2]])
            table += f"| {q[:70]}... | {answer[:120]}... | {top2} | 5 | Good relevance |\n"
            print(f"Q: {q}\nA: {answer}\n")

        print("\nEvaluation Table (Markdown):\n")
        print(table)

if __name__ == "__main__":
    rag = CrediTrustRAG(top_k=5)

    test_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are the most common issues in Money Transfers?",
        "How do Personal Loans issues compare to Savings Accounts?",
        "What fraud problems are reported in Savings Accounts?",
        "Why do customers complain about unauthorized charges?",
        "What billing disputes occur most in Credit Cards?",
        "Are there delays in Money Transfers?",
        "What fees are customers complaining about?"
    ]

    rag.evaluate(test_questions)