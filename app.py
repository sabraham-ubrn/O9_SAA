from datetime import datetime
import os
from pathlib import Path
import streamlit as st
from rag_pipeline import create_or_load_vector_store, get_qa_chain

# Set global styles
font_color = "#000000"  # white font
##ffffff
bg_color = "#000000"    # black background

CONFIG_DIR = "Config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.toml")

st.set_page_config(page_title="O9 Smart Chat Assistant", layout="centered")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    button[title="View app in Streamlit Community Cloud"] {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Add this to the top of your app.py after `st.set_page_config(...)`
st.markdown(f"""
    <style>
        .time_based_title {{
            font-size: 24px;
            font-weight: bold;
            color: {font_color};
            text-align: left;
        }}
        .time_based_caption {{
            font-size: 16px;
            color: {font_color};
            text-align: left;
        }}
    </style>
    <div class="time_based_title">
        💬 O9 Chat Assistant --> What can I help you?
    </div>
    <div class="time_based_caption">
        Ask questions about the O9 Training PDF (powered by RAG + Llama 3)
    </div>
            
""", unsafe_allow_html=True)

#st.caption("")
#background-color:{bg_color};background-color:{bg_color};
# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Load PDF and build chain
pdf_path = Path("./data/o9_Onboarding.pdf")

vectordb, qa_chain = None, None
if pdf_path.exists() and pdf_path.stat().st_size > 0:
    try:
        vectordb = create_or_load_vector_store(pdf_path)
        qa_chain = get_qa_chain(vectordb)
    except Exception as e:
        st.error(f"Failed to initialize PDF processing: {e}")
        st.stop()
else:
    st.warning("PDF not found or empty. Please upload a valid file.")
    st.stop()

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if user_input := st.chat_input("Ask anything about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.message_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Analyzing your question..."):
        try:
            response = qa_chain.invoke({"query": user_input})
            # Debugging the structure of the response
            #st.write(response)  # This will display the full response to the Streamlit interface

            answer = response["result"]
            sources = response.get("source_documents", [])

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.message_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

                if sources:
                    with st.expander("📄 Source Excerpts"):
                        for doc in sources:
                            st.markdown(doc.page_content[:500] + "...")
        except Exception as e:
            error_msg = f"Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.message_history.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
