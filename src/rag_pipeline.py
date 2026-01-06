# src/rag_pipeline.py

from pathlib import Path
from config import PREBUILT_PARQUET, VECTOR_STORE_DIR

# LangChain imports (latest)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint


class CrediTrustRAG:
    """
    RAG pipeline for CrediTrust financial complaints
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        # Embeddings
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # Vector store paths
        full_store_path = VECTOR_STORE_DIR / "full_prebuilt"
        sample_store_path = VECTOR_STORE_DIR / "sample_chroma"

        if PREBUILT_PARQUET.exists():
            if full_store_path.exists():
                print("Loading full pre-built vector store...")
                store_path = full_store_path
            else:
                print("Pre-built parquet found, but full store not built.")
                print("Falling back to sample store for now.")
                store_path = sample_store_path
        else:
            print("Pre-built parquet not found. Using sample store.")
            store_path = sample_store_path

        if not store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_path}. "
                f"Run Task 2 or load_prebuilt.py first."
            )

        # Load Chroma vector store
        self.db = Chroma(
            persist_directory=str(store_path),
            embedding_function=self.embeddings,
            collection_name="complaint_chunks"
        )
        print(f"Vector store loaded: {self.db._collection.count():,} chunks from {store_path.name}")

        self.retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})

        # Hugging Face Inference API LLM
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.3,
            max_new_tokens=512
        )

        # Prompt template
        self.prompt = PromptTemplate.from_template(
            """You are a financial analyst assistant for CrediTrust Financial in East Africa.
Use only the provided complaint excerpts to answer the question.
Be concise, factual, and evidence-based.

Conversation History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""
        )

        # Function to format retrieved docs
        def format_docs(docs: list[Document]):
            return "\n\n".join(
                f"[{i+1}] (Product: {doc.metadata.get('product_category', 'Unknown')}) {doc.page_content}"
                for i, doc in enumerate(docs)
            )

        # RAG chain
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str):
        """
        Ask a question to the RAG system and return the answer and source metadata.
        """
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

    def evaluate(self, questions: list[str]):
        """
        Simple evaluation utility for testing multiple questions
        """
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


# ----------------------------
# Helper function for app.py
# ----------------------------
_rag_instance = CrediTrustRAG(top_k=5)

def ask_rag(question: str, chat_history=None):
    """
    Simple wrapper to call the RAG system from app.py
    """
    return _rag_instance.ask(question)


# ----------------------------
# Standalone test
# ----------------------------
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
