import streamlit as st
from hugchat import hugchat
from transformers import pipeline
from pypdf import PdfReader
import io

st.set_page_config(page_title="AI Finance Chatbot👾")
st.header('AI Finance Chatbot👾')

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    file_contents = uploaded_file.read()
    remote_file_bytes = io.BytesIO(file_contents)
    pdfdoc_remote = PdfReader(remote_file_bytes)

    pdf_text = ""

    for i in range(len(pdfdoc_remote.pages)):
        print(i)
        page = pdfdoc_remote.pages[i]
        page_content = page.extract_text()
        pdf_text += page_content

    print(pdf_text)

    nlp = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
    )

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the document!"}
    ]

@st.cache_resource(show_spinner=False)
def generate_answer(prompt, pdf_text):
    context = pdf_text
    question = prompt
    question_set = {"context": context, "question": question}
    results = nlp(question_set)
    print("\nAnswer: " + results["answer"])
    return results["answer"]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt, pdf_text)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history

