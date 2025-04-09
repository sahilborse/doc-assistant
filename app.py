import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Chat PDF", page_icon="💁", layout="wide")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, just say, "answer is not available in the context". Don't guess.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    if st.session_state["theme"] == "Dark":
        st.markdown(f"""
            <div style='background-color:#1f1f1f; color:white; padding:15px; border-radius:10px; margin-top:10px;'>
                <strong>Reply:</strong><br>{response["output_text"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color:#f1f1f1; color:black; padding:15px; border-radius:10px; margin-top:10px;'>
                <strong>Reply:</strong><br>{response["output_text"]}
            </div>
        """, unsafe_allow_html=True)

def apply_custom_css(theme):
    background_url = "https://www.transparenttextures.com/patterns/white-wall.png" if theme == "Light" else "https://www.transparenttextures.com/patterns/dark-mosaic.png"
    text_color = "black" if theme == "Light" else "white"
    bg_color = "#ffffffcc" if theme == "Light" else "#000000cc"

    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{background_url}");
            background-size: cover;
            background-repeat: repeat;
        }}
        .title-style {{
            color: {text_color};
            text-shadow: 1px 1px 2px #000;
        }}
        .block-container {{
            background-color: {bg_color};
            border-radius: 12px;
            padding: 2rem;
            margin-top: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

def light_dark_toggle():
    toggle_js = """
    <script>
        const themeIcon = document.getElementById("theme-toggle");
        themeIcon.addEventListener("click", () => {{
            fetch("/?toggle_theme=1").then(() => window.location.reload());
        }});
    </script>
    """

    icon_url = "https://cdn-icons-png.flaticon.com/512/4814/4814316.png" if st.session_state.get("theme", "Light") == "Light" else "https://cdn-icons-png.flaticon.com/512/7283/7283459.png"

    # st.markdown(f"""
    #     <style>
    #     .theme-toggle {{
    #         animation: float 3s ease-in-out infinite;
    #         position: fixed;
    #         bottom: 20px;
    #         right: 20px;
    #         width: 60px;
    #         cursor: pointer;
    #         z-index: 9999;
    #     }}
    #     @keyframes float {{
    #         0% {{ transform: translatey(0px); }}
    #         50% {{ transform: translatey(-10px); }}
    #         100% {{ transform: translatey(0px); }}
    #     }}
    #     </style>
    #     <img id="theme-toggle" src="{icon_url}" class="theme-toggle">
    #     {toggle_js}
    # """, unsafe_allow_html=True)

def main():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    if st.query_params.get("toggle_theme"):
        st.session_state["theme"] = "Dark" if st.session_state["theme"] == "Light" else "Light"

    apply_custom_css(st.session_state["theme"])
    light_dark_toggle()

    st.markdown("<h1 class='title-style'>💬 Chat with PDF using Gemini</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center'>
            <img src='https://media.giphy.com/media/QBd2kLB5qDmysEXre9/giphy.gif' width='250'>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .dark-input input {
            background-color: #1f1f1f !important;
            color: white !important;
            border: 1px solid #555 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state["theme"] == "Dark":
        user_question = st.text_input("🔍 Ask a Question from the PDF Files", key="dark_input", label_visibility="visible")
        st.markdown("<div class='dark-input'></div>", unsafe_allow_html=True)
    else:
        user_question = st.text_input("🔍 Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/file-search-concept-illustration_114360-4287.jpg", use_container_width=True)
        st.title("📚 Menu")
        pdf_docs = st.file_uploader("📎 Upload your PDF Files", accept_multiple_files=True)
        if st.button("🚀 Submit & Process"):
            with st.spinner("⏳ Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done")

if __name__ == "__main__":
    main()
