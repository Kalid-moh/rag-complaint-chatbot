# rag-complaint-chatbot

**Intelligent Complaint Analysis for Financial Services**  
A Retrieval-Augmented Generation (RAG) powered internal chatbot that turns raw customer complaint data into actionable insights for CrediTrust Financial.

## Project Goal

CrediTrust Financial is a fast-growing digital finance company serving East African markets with products including **Credit Cards**, **Personal Loans**, **Savings Accounts**, and **Money Transfers**. With over 500,000 users and thousands of monthly complaints, internal teams (Product, Support, Compliance) struggle to quickly identify emerging issues from unstructured narrative text.
This project builds an **internal AI tool** that enables non-technical stakeholders (e.g., Product Manager Asha) to ask plain-English questions like:

- “Why are people unhappy with Credit Cards?”
- “What are the most common fraud issues in Savings Accounts?”
  and receive **concise, evidence-backed answers in seconds**, sourced directly from real customer complaints.

### Key Performance Indicators (KPIs)

- Reduce time to identify major complaint trends from **days to minutes**
- Empower non-technical teams to get insights without data analysts
- Shift from reactive to **proactive** issue resolution using real-time feedback

## Folder Structure

rag-complaint-chatbot/
├── .venv/ # Virtual environment (gitignored)
├── data/
│ ├── raw/ # Raw CFPB complaints.csv.zip
│ ├── processed/ # filtered_complaints.csv (Task 1 output)
│ └── complaint_embeddings.parquet # Pre-built embeddings (provided)
├── vector_store/
│ ├── sample_chroma/ # Task 2 sample vector store
│ └── full_prebuilt/ # Task 3–4 full-scale vector store
├── notebooks/
│ ├── eda.ipynb # Task 1 EDA & preprocessing notebook
│ └── eda_task2.ipynb # Task 2 vector store building notebook
├── src/
│ ├── config.py # Path and configuration constants
│ ├── preprocessor.py # Task 1: EDA & preprocessing script
│ ├── vector_store_builder.py # Task 2: Sampling, chunking, indexing
│ ├── load_prebuilt.py # Load pre-built parquet into Chroma
│ └── rag_pipeline.py # Task 3: RAG core logic & evaluation
├── app.py # Task 4: Gradio UI (to be implemented)
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore

## Prerequisites

- **Python 3.12.10 - 3.12.11** (recommended — full wheel support for all packages)
- Git
- At least 8 GB RAM (16 GB recommended for full pre-built store)
- Internet connection (for downloading CFPB data and Hugging Face models)

> Note: Avoid Python 3.13 or 3.14 — limited wheel support causes installation errors.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Kalid-moh/rag-complaint-chatbot.git
cd rag-complaint-chatbot

pip install uv
uv venv
# Activate
.venv\Scripts\activate      # Windows PowerShell/Command Prompt
# or
source .venv/bin/activate   # Git Bash / Linux / macOS

py -3.12 -m venv .venv
.venv\Scripts\activate

uv pip install -r requirements.txt
or
pip install -r requirements.txt

uv run src/preprocessor.py
or
python src/preprocessor.py

uv run src/vector_store_builder.py
```
