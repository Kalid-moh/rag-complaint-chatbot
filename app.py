import gradio as gr
import time
from src.rag_pipeline import CrediTrustRAG

rag = CrediTrustRAG(top_k=5)

def format_history(chat_history):
    return "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history]
    )

def stream_answer(question, chat_history):
    if not question.strip():
        yield chat_history, ""

    history_text = format_history(chat_history)

    answer, sources = rag.ask(question, history_text)

    streamed = ""
    for token in answer.split():
        streamed += token + " "
        time.sleep(0.03)
        yield chat_history + [(question, streamed)], ""

    # Append sources
    sources_md = "\n\n### 📚 Sources Used\n"
    for i, s in enumerate(sources, 1):
        sources_md += (
            f"**{i}. Product:** {s['product_category']}  \n"
            f"**Complaint ID:** {s['complaint_id']}  \n"
            f"> {s['text_preview']}\n\n"
        )

    final_answer = streamed + sources_md
    yield chat_history + [(question, final_answer)], ""

def clear_chat():
    return [], ""

with gr.Blocks(title="CrediTrust RAG Assistant") as demo:
    gr.Markdown("# 🤖 CrediTrust Complaint Analysis Assistant")
    gr.Markdown(
        "Ask questions about customer complaints. "
        "Answers are **retrieval-augmented and source-verified**."
    )

    chatbot = gr.Chatbot(height=420)
    question_box = gr.Textbox(
        label="Ask a Question",
        placeholder="e.g. Why are customers unhappy with credit cards?",
        lines=2
    )

    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    ask_btn.click(
        stream_answer,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, question_box]
    )

demo.launch()
