# app.py
import gradio as gr
import time
# Inside src/rag_pipeline.py
# src/rag_pipeline.py
from src.settings import PREBUILT_PARQUET, VECTOR_STORE_DIR




# ----------------------------
# Streaming generator
# ----------------------------
def stream_answer(question, chat_history):
    """
    Streams tokens from RAG while keeping chat history
    """
    if not question.strip():
        yield chat_history, ""

    # Call RAG system
    answer, sources = ask_rag(question, chat_history)

    streamed_text = ""
    # Stream word by word (simulate typing)
    for token in answer.split():
        streamed_text += token + " "
        time.sleep(0.03)  # Adjust speed as needed
        # Update chat with current partial answer
        yield chat_history + [(question, streamed_text)], ""

    # Append sources at the end
    if sources:
        sources_text = "\n\n### 📚 Sources Used:\n"
        for i, src in enumerate(sources, 1):
            sources_text += (
                f"**Source {i}:** Complaint ID: {src['complaint_id']}, "
                f"Product: {src['product_category']}\n"
                f"{src['text_preview']}\n\n"
            )
        final_answer = streamed_text + sources_text
    else:
        final_answer = streamed_text

    # Final yield with sources appended
    yield chat_history + [(question, final_answer)], ""


# ----------------------------
# Clear chat
# ----------------------------
def clear_chat():
    return [], ""


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="RAG Chat Assistant") as demo:
    gr.Markdown("# 🤖 RAG-Powered Chat Assistant")
    gr.Markdown(
        "Ask questions and get **source-backed answers** from your documents."
    )

    chatbot = gr.Chatbot(height=400)
    question_box = gr.Textbox(
        label="Your Question",
        placeholder="Ask something...",
        lines=2
    )

    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    # Button click events
    ask_btn.click(
        stream_answer,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, question_box]
    )

# Launch the app
demo.launch()
